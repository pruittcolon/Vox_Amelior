import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: '/home/pruittcolon/Desktop/Nemo_Server/tests/playwright',
  testMatch: ['nexus-visualization-smoke.spec.ts'],
  fullyParallel: false,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'off',
    screenshot: 'only-on-failure'
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    }
  ],
  webServer: {
    command: 'python3 -m http.server 4173 --directory /home/pruittcolon/Desktop/Nemo_Server/frontend',
    url: 'http://127.0.0.1:4173/',
    reuseExistingServer: true,
    timeout: 120 * 1000
  }
});
