# Equipment Spec Sheet -- Line 6"-HC-2201 (P&ID-14-B)

**Document owner:** Process Engineering
**Applies to:** P&ID-14-B, Line 6"-HC-2201

| Tag | Equipment | Required type | Notes |
|-----|-----------|----------------|-------|
| PSV-2201 | Pressure safety valve | Spring-loaded relief valve | Set pressure per relief study RS-2201, separator overpressure protection |
| FV-2202 | Flow control valve | Control valve, pneumatic actuator, fail-closed on air loss | Modulates flow to downstream separator; must fail closed per HAZOP action item HZ-2201-07 |
| PT-2203 | Pressure transmitter | Field-mounted pressure transmitter | 4-20mA signal to DCS, alarm high at 18 barg |

Any deviation from the required type above (e.g. a manual valve installed
where a control valve is specified) must be raised to Process Engineering
before the line is returned to service -- a manual gate valve cannot provide
the fail-closed, remotely-modulated action the HAZOP action item requires.
