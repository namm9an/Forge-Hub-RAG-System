import type { Config } from 'jest';

const config: Config = {
  testEnvironment: 'jsdom',
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest'
  },
  moduleNameMapper: {
    '^@common/(.*)$': '<rootDir>/../../packages/common/src/$1'
  }
};

export default config;

