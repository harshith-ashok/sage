// marked-katex-extension ships raw .ts as its "types" entry (not a .d.ts),
// which makes vue-tsc fully type-check that file against this project's
// strict compiler options and fail on an unused-parameter lint issue in
// their source. Redirected here (see tsconfig.app.json "paths") to a
// minimal ambient declaration instead of pulling in their source file.
import type { MarkedExtension } from "marked";

interface KatexOptions {
  throwOnError?: boolean;
  nonStandard?: boolean;
  displayMode?: boolean;
}

declare function markedKatex(options?: KatexOptions): MarkedExtension;
export default markedKatex;
