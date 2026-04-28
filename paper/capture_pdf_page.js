const { chromium } = require('playwright-core');
const path = require('path');

async function main() {
  const pdfPath = process.argv[2];
  const pageNum = process.argv[3];
  const outPath = process.argv[4];
  if (!pdfPath || !pageNum || !outPath) {
    throw new Error('Usage: node capture_pdf_page.js <pdfPath> <pageNum> <outPath>');
  }

  const browser = await chromium.launch({
    executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: true,
  });

  const page = await browser.newPage({
    viewport: { width: 1600, height: 2200 },
    deviceScaleFactor: 2,
  });

  const url = 'file:///' + pdfPath.replace(/\\/g, '/').replace(/^([A-Za-z]):/, '$1:') + '#page=' + pageNum;
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForTimeout(2500);
  await page.keyboard.press('Home').catch(() => {});
  const steps = Math.max(0, Number(pageNum) - 1);
  for (let i = 0; i < steps; i += 1) {
    await page.keyboard.press('PageDown');
    await page.waitForTimeout(350);
  }
  await page.screenshot({ path: outPath, fullPage: true });
  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
