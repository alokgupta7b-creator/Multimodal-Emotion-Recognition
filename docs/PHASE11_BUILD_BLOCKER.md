# Phase 11 Vite Build Blocker Investigation

## Result

The production-build blocker was confirmed as a Windows Enterprise Code
Integrity policy decision, not a missing or incorrect Rollup package. The
build now succeeds without disabling security controls, bypassing policy, or
changing dependencies.

## Evidence

Environment:

```text
Node.js v24.21.0
npm 11.19.0
Vite 6.4.6
Rollup 4.68.2
@rollup/rollup-win32-x64-msvc 4.68.2
```

The installed dependency tree contains the correct platform package:

```text
vite@6.4.6
└─ rollup@4.68.2
   └─ @rollup/rollup-win32-x64-msvc@4.68.2
```

The exact earlier Node error was:

```text
Error: An Application Control policy has blocked this file.
\?\C:\...\frontend\node_modules\@rollup\rollup-win32-x64-msvc\rollup.win32-x64-msvc.node
code: 'ERR_DLOPEN_FAILED'
```

Windows Code Integrity Operational events corroborated it:

- Event 3077 at `2026-11-10 20:03:22`: the Node process attempted to load
  the Rollup native binary, which did not meet Enterprise signing-level
  requirements or violated Code Integrity policy
  `0283ac0f-fff1-49ae-ada1-8a933130cad6`.
- Event 3033 at the same time: the binary did not meet Enterprise signing
  level requirements.

The native module is now directly loadable:

```text
node -e "require('@rollup/rollup-win32-x64-msvc')"
```

returned the expected Rollup exports (`parse`, `parseAsync`, and hash
functions).

## Supported fallback

Rollup's installed `dist/native.js` explicitly identifies
`@rollup/wasm-node` as the supported fallback for unsupported native
platforms. `npm view @rollup/wasm-node version` returned `4.68.2`.
It was not installed or substituted because the current Windows x64 native
package is correct and the production build succeeds. No fallback dependency
change was justified.

## Validation

Commands and results:

```text
cd frontend
npx tsc -b
```

Passed.

```text
npm run build
```

Passed:

```text
vite v6.4.6 building for production...
1578 modules transformed.
built in 5.10s
```

Frontend tests: no test script or frontend test runner is currently defined
in `frontend/package.json`; `npm run` confirms only `dev`, `build`, and
`preview`. No frontend test command was fabricated.

```text
cd ..
python -m pytest api -q
```

Passed: **9 passed, 3 warnings**.

```text
python -m pytest -q
```

Passed: **216 passed, 6 warnings**.

Runtime dependency audit:

```text
npm audit --omit=dev
```

Passed with **0 vulnerabilities**.

## Actions not taken

- Windows security and Enterprise policy were not disabled or bypassed.
- No blind reinstall or broad dependency upgrade was performed.
- `package-lock.json` was preserved.
- No model, dataset, backend, or Phase 1-10 artifact was modified.
- No Phase 12 work was started.

## Safe build guidance

The build is currently verified on this Windows environment. If the policy
block recurs, use an approved development/CI environment where the
organization permits the Rollup native binary, or use the matching
`@rollup/wasm-node` fallback through an explicitly reviewed toolchain change.
Do not weaken Code Integrity policy to run the build.
