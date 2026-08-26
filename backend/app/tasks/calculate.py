"""Phase 5: parameterized calculation tool. sympy parses and evaluates the
expression (with unit tracking via sympy.physics.units); numpy vectorizes
when a variable is given as an array of values. The caller (an LLM, later
phases) supplies the expression and values — it never computes the
arithmetic itself, sympy/numpy always do, so the result is exact/traceable
rather than a model's guess.
"""

import numpy as np
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations
from sympy.physics import units
from sympy.physics.units import convert_to

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_BASE_UNIT_NAMES = {
    "meter": units.meter,
    "second": units.second,
    "kilogram": units.kilogram,
    "gram": units.gram,
    "newton": units.newton,
    "pascal": units.pascal,
    "bar": units.bar,
    "watt": units.watt,
    "joule": units.joule,
    "kelvin": units.kelvin,
    "minute": units.minute,
    "hour": units.hour,
    "day": units.day,
    "week": units.day * 7,
    "year": units.year,
    # sympy.physics.units has no native "month" — a standard 1/12-year
    # engineering approximation, not a precise calendar month.
    "month": units.year / 12,
}

_UNIT_SYMBOLS = {
    "m": units.meter,
    "s": units.second,
    "kg": units.kilogram,
    "g": units.gram,
    "N": units.newton,
    "Pa": units.pascal,
    "W": units.watt,
    "J": units.joule,
    "K": units.kelvin,
}

# Plural forms ("years", "months", "meters", ...) matter here specifically
# because this expression parser treats an unrecognized word as several
# multiplied one-letter symbols instead of raising an error — verified
# live: "10 years / 1 year" silently parsed as
# 10*second*a**2*e**2*r**2*y**2 instead of failing, because only the
# singular "year" was registered. Every base unit name gets its plural
# alias here so that failure mode doesn't require enumerating every
# expression a caller might phrase in plural.
_UNIT_NAMES = {
    **_BASE_UNIT_NAMES,
    **{f"{name}s": unit for name, unit in _BASE_UNIT_NAMES.items()},
    **_UNIT_SYMBOLS,
}

_MATH_NAMES = {
    "pi": sympy.pi,
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "exp": sympy.exp,
    "log": sympy.log,
    "Abs": sympy.Abs,
}


class CalculationError(Exception):
    pass


def calculate(
    expression: str,
    variables: dict[str, float | list[float]] | None = None,
    convert_to_unit: str | None = None,
) -> dict:
    variables = variables or {}
    array_vars = {k: v for k, v in variables.items() if isinstance(v, list)}
    scalar_vars = {k: v for k, v in variables.items() if not isinstance(v, list)}

    local_dict: dict = dict(_UNIT_NAMES)
    local_dict.update(_MATH_NAMES)
    var_symbols = {name: sympy.Symbol(name) for name in variables}
    local_dict.update(var_symbols)

    try:
        parsed = parse_expr(expression, local_dict=local_dict, transformations=_TRANSFORMATIONS)
    except Exception as exc:
        raise CalculationError(f"Could not parse expression {expression!r}: {exc}") from exc

    # implicit_multiplication_application silently turns any unrecognized
    # word into several multiplied one-letter symbols instead of raising —
    # verified live: "10 years / 1 year" (a missing unit at the time)
    # parsed "cleanly" into 10*second*a**2*e**2*r**2*y**2, a wrong answer
    # with no error at all. Anything left in free_symbols that isn't a
    # declared variable is exactly that failure mode, so it's rejected here
    # instead of being allowed to produce a silently-nonsensical result.
    unexpected = parsed.free_symbols - set(var_symbols.values())
    if unexpected:
        names = sorted(str(s) for s in unexpected)
        raise CalculationError(
            f"Unrecognized name(s) in expression: {', '.join(names)}. "
            "Not a known unit/constant and not declared in `variables`."
        )

    steps = [f"Parsed: {parsed}"]

    if array_vars:
        if scalar_vars:
            parsed = parsed.subs({var_symbols[k]: sympy.Float(v) for k, v in scalar_vars.items()})
            steps.append(f"Substituted scalar values {scalar_vars}: {parsed}")
        array_names = list(array_vars)
        fn = sympy.lambdify([var_symbols[n] for n in array_names], parsed, modules="numpy")
        arrays = [np.array(array_vars[n], dtype=float) for n in array_names]
        result_array = fn(*arrays)
        steps.append(f"Vectorized over {array_names} with numpy ({len(arrays[0])} values each)")
        return {
            "expression": expression,
            "steps": steps,
            "result": [float(v) for v in np.atleast_1d(result_array)],
            "is_array": True,
        }

    substituted = parsed.subs({var_symbols[k]: sympy.Float(v) for k, v in scalar_vars.items()})
    if scalar_vars:
        steps.append(f"Substituted {scalar_vars}: {substituted}")

    result = sympy.simplify(substituted)
    steps.append(f"Simplified: {result}")

    if convert_to_unit:
        target = local_dict.get(convert_to_unit)
        if target is None:
            raise CalculationError(f"Unknown target unit {convert_to_unit!r}")
        result = convert_to(result, target).evalf()
        steps.append(f"Converted to {convert_to_unit}: {result}")

    is_plain_number = not result.free_symbols and not result.has(units.Quantity)
    numeric = float(result.evalf()) if is_plain_number else None

    return {
        "expression": expression,
        "steps": steps,
        "result": str(result),
        "numeric": numeric,
        "is_array": False,
    }
