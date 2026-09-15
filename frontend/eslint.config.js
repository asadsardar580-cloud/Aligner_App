import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  // dist/ and ssr_out/ are BUILD ARTIFACTS, not source. ssr_out/ was being
  // linted and produced half of every reported problem — errors about
  // generated bundle code that no one can act on, drowning the real ones.
  globalIgnores(['dist', 'ssr_out', 'ssr_out2']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    rules: {
      // A leading underscore marks a parameter that is deliberately unused but
      // is part of a callback's documented signature — ToothGizmo passes
      // (clinical, matrixRowMajor) to onChange/onCommit, and dropping the
      // second name would hide the contract rather than document it.
      'no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },
  {
    // Node scripts, not browser code: the SSR smoke harness and the
    // cross-language kinematics verifier both run under node and use `process`.
    files: ['ssr_smoke.jsx', 'verify-kinematics.mjs'],
    languageOptions: { globals: globals.node },
  },
])
