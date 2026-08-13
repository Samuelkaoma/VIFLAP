/**
 * ESLint configuration.
 *
 * The `.eslintrc` format rather than flat config, because the pinned ESLint is
 * 8.x and flat config is not its default. Matching the installed major keeps
 * `npm run lint` working from a clean checkout without a resolution step.
 *
 * The rules that matter here are not stylistic. This interface renders evidence
 * about people, and two of the errors below are the ones that would actually
 * cause harm:
 *
 * `no-floating-promises` — an unawaited request whose rejection is never
 * handled leaves the previous result on screen with no error shown. The reader
 * sees a stale comparison and no indication that anything failed.
 *
 * `react-hooks/exhaustive-deps` — a memoised client that does not rebuild when
 * the case reference changes issues requests bound to the previous case. That
 * is a governance failure, and it looks like nothing at all.
 */

module.exports = {
  root: true,
  env: { browser: true, es2022: true, node: true },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module',
    ecmaFeatures: { jsx: true },
    // Type-aware linting. Costs a slower run and buys the rules below, which
    // cannot be expressed without type information.
    project: ['./tsconfig.json'],
    tsconfigRootDir: __dirname,
  },
  plugins: ['@typescript-eslint', 'react-hooks'],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:@typescript-eslint/recommended-requiring-type-checking',
  ],
  settings: { react: { version: '18.3' } },
  rules: {
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'error',
    '@typescript-eslint/no-floating-promises': 'error',
    '@typescript-eslint/no-misused-promises': 'error',
    // Unused arguments are permitted when prefixed, which is how an event
    // handler declares that it ignores its event.
    '@typescript-eslint/no-unused-vars': [
      'error',
      { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
    ],
    // `any` erases the guarantee that a result carries its prior, which is the
    // property the types exist to enforce.
    '@typescript-eslint/no-explicit-any': 'error',
  },
  ignorePatterns: ['dist', 'node_modules', 'vite.config.ts', '.eslintrc.cjs'],
};
