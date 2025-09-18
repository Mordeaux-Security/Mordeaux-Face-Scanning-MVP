import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

export interface EnvConfig {
  NODE_ENV: 'development' | 'production' | 'test';
  LOG_LEVEL: string;
  PORT: number;
  DATABASE_URL: string;
  REDIS_URL: string;
  RABBITMQ_URL: string;
  MINIO_ENDPOINT: string;
  MINIO_ACCESS_KEY: string;
  MINIO_SECRET_KEY: string;
  JWT_SECRET: string;
  VECTOR_DB_URL: string;
}

function getEnvVar(key: keyof EnvConfig, defaultValue?: string): string {
  const value = process.env[key] || defaultValue;
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
  return value;
}

function getEnvNumber(key: keyof EnvConfig, defaultValue?: number): number {
  const value = process.env[key];
  if (!value && defaultValue === undefined) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
  return value ? parseInt(value, 10) : defaultValue!;
}

export const env: EnvConfig = {
  NODE_ENV: (process.env.NODE_ENV as EnvConfig['NODE_ENV']) || 'development',
  LOG_LEVEL: getEnvVar('LOG_LEVEL', 'info'),
  PORT: getEnvNumber('PORT', 3000),
  DATABASE_URL: getEnvVar('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/mordeaux'),
  REDIS_URL: getEnvVar('REDIS_URL', 'redis://localhost:6379'),
  RABBITMQ_URL: getEnvVar('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672'),
  MINIO_ENDPOINT: getEnvVar('MINIO_ENDPOINT', 'http://localhost:9000'),
  MINIO_ACCESS_KEY: getEnvVar('MINIO_ACCESS_KEY', 'minioadmin'),
  MINIO_SECRET_KEY: getEnvVar('MINIO_SECRET_KEY', 'minioadmin'),
  JWT_SECRET: getEnvVar('JWT_SECRET', 'dev-secret-key'),
  VECTOR_DB_URL: getEnvVar('VECTOR_DB_URL', 'http://localhost:8080')
};

export function validateEnv(): void {
  // This will throw if any required env vars are missing
  Object.keys(env).forEach(key => {
    if (env[key as keyof EnvConfig] === undefined) {
      throw new Error(`Environment validation failed: ${key} is undefined`);
    }
  });
}
