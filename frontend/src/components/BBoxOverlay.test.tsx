/**
 * BBoxOverlay Visual Tests - Phase 6
 * ===================================
 * 
 * Tests BBox alignment accuracy across varied aspect ratios.
 * Tolerance: ±2% alignment accuracy.
 */

/**
 * Test Cases for BBox Overlay
 * ============================
 * 
 * These test various image aspect ratios and BBox positions to ensure
 * consistent rendering within ±2% tolerance.
 */

export interface BBoxTestCase {
  name: string;
  description: string;
  imageDimensions: { width: number; height: number };
  bbox: [number, number, number, number];
  expectedCSS: {
    left: string;
    top: string;
    width: string;
    height: string;
  };
  tolerance: number; // Percentage tolerance
}

/**
 * Test cases for varied aspect ratios
 */
export const BBOX_TEST_CASES: BBoxTestCase[] = [
  {
    name: 'Square Image - Center BBox',
    description: 'BBox centered in square 1024x1024 image',
    imageDimensions: { width: 1024, height: 1024 },
    bbox: [412, 412, 200, 200],
    expectedCSS: {
      left: '40.23%',
      top: '40.23%',
      width: '19.53%',
      height: '19.53%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Landscape Image - Top-Left BBox',
    description: 'BBox in top-left corner of landscape 1920x1080 image',
    imageDimensions: { width: 1920, height: 1080 },
    bbox: [100, 100, 300, 300],
    expectedCSS: {
      left: '5.21%',
      top: '9.26%',
      width: '15.62%',
      height: '27.78%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Portrait Image - Bottom-Right BBox',
    description: 'BBox in bottom-right of portrait 1080x1920 image',
    imageDimensions: { width: 1080, height: 1920 },
    bbox: [680, 1520, 300, 300],
    expectedCSS: {
      left: '62.96%',
      top: '79.17%',
      width: '27.78%',
      height: '15.62%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Small Image - Full Face',
    description: 'Large BBox covering most of small 256x256 image',
    imageDimensions: { width: 256, height: 256 },
    bbox: [28, 28, 200, 200],
    expectedCSS: {
      left: '10.94%',
      top: '10.94%',
      width: '78.12%',
      height: '78.12%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Ultra-Wide - Center BBox',
    description: 'BBox in center of ultra-wide 3440x1440 image',
    imageDimensions: { width: 3440, height: 1440 },
    bbox: [1520, 570, 400, 300],
    expectedCSS: {
      left: '44.19%',
      top: '39.58%',
      width: '11.63%',
      height: '20.83%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Vertical Mobile - Top BBox',
    description: 'BBox at top of vertical mobile 750x1334 image',
    imageDimensions: { width: 750, height: 1334 },
    bbox: [175, 200, 400, 400],
    expectedCSS: {
      left: '23.33%',
      top: '15.00%',
      width: '53.33%',
      height: '30.00%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Edge Case - Minimum Size',
    description: 'Minimum viable BBox (1% of image)',
    imageDimensions: { width: 1000, height: 1000 },
    bbox: [400, 400, 10, 10],
    expectedCSS: {
      left: '40.00%',
      top: '40.00%',
      width: '1.00%',
      height: '1.00%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Edge Case - Edge BBox',
    description: 'BBox touching image edge',
    imageDimensions: { width: 1024, height: 1024 },
    bbox: [0, 0, 200, 200],
    expectedCSS: {
      left: '0.00%',
      top: '0.00%',
      width: '19.53%',
      height: '19.53%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Edge Case - Maximum Size',
    description: 'BBox covering almost entire image',
    imageDimensions: { width: 1024, height: 1024 },
    bbox: [10, 10, 1004, 1004],
    expectedCSS: {
      left: '0.98%',
      top: '0.98%',
      width: '98.05%',
      height: '98.05%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Varied Aspect - 4:3 Image',
    description: 'Classic 4:3 aspect ratio 1600x1200 image',
    imageDimensions: { width: 1600, height: 1200 },
    bbox: [600, 400, 400, 400],
    expectedCSS: {
      left: '37.50%',
      top: '33.33%',
      width: '25.00%',
      height: '33.33%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Varied Aspect - 16:9 Image',
    description: 'Standard 16:9 aspect ratio 1920x1080 image',
    imageDimensions: { width: 1920, height: 1080 },
    bbox: [760, 340, 400, 400],
    expectedCSS: {
      left: '39.58%',
      top: '31.48%',
      width: '20.83%',
      height: '37.04%',
    },
    tolerance: 2,
  },
  
  {
    name: 'Varied Aspect - 21:9 Ultrawide',
    description: 'Ultrawide 21:9 aspect ratio 2560x1080 image',
    imageDimensions: { width: 2560, height: 1080 },
    bbox: [1080, 340, 400, 400],
    expectedCSS: {
      left: '42.19%',
      top: '31.48%',
      width: '15.62%',
      height: '37.04%',
    },
    tolerance: 2,
  },
];

/**
 * Validate BBox conversion accuracy
 * Returns true if within tolerance
 */
export function validateBBoxAccuracy(
  actual: { left: string; top: string; width: string; height: string },
  expected: { left: string; top: string; width: string; height: string },
  tolerance: number
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  
  // Parse percentages
  const actualLeft = parseFloat(actual.left);
  const actualTop = parseFloat(actual.top);
  const actualWidth = parseFloat(actual.width);
  const actualHeight = parseFloat(actual.height);
  
  const expectedLeft = parseFloat(expected.left);
  const expectedTop = parseFloat(expected.top);
  const expectedWidth = parseFloat(expected.width);
  const expectedHeight = parseFloat(expected.height);
  
  // Check each dimension
  if (Math.abs(actualLeft - expectedLeft) > tolerance) {
    errors.push(`Left: ${actualLeft}% vs ${expectedLeft}% (tolerance: ${tolerance}%)`);
  }
  
  if (Math.abs(actualTop - expectedTop) > tolerance) {
    errors.push(`Top: ${actualTop}% vs ${expectedTop}% (tolerance: ${tolerance}%)`);
  }
  
  if (Math.abs(actualWidth - expectedWidth) > tolerance) {
    errors.push(`Width: ${actualWidth}% vs ${expectedWidth}% (tolerance: ${tolerance}%)`);
  }
  
  if (Math.abs(actualHeight - expectedHeight) > tolerance) {
    errors.push(`Height: ${actualHeight}% vs ${expectedHeight}% (tolerance: ${tolerance}%)`);
  }
  
  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Run all test cases
 * 
 * Usage in test file:
 * 
 * import { BBOX_TEST_CASES, validateBBoxAccuracy } from './BBoxOverlay.test';
 * 
 * describe('BBoxOverlay', () => {
 *   BBOX_TEST_CASES.forEach(testCase => {
 *     it(testCase.name, () => {
 *       const result = bboxToPercentages(testCase.bbox, testCase.imageDimensions);
 *       const validation = validateBBoxAccuracy(result, testCase.expectedCSS, testCase.tolerance);
 *       expect(validation.valid).toBe(true);
 *     });
 *   });
 * });
 */

/**
 * Generate visual test HTML
 * For manual visual inspection in browser
 */
export function generateVisualTestHTML(testCases: BBoxTestCase[]): string {
  return `
<!DOCTYPE html>
<html>
<head>
  <title>BBox Overlay Visual Tests</title>
  <style>
    body {
      font-family: system-ui;
      padding: 20px;
      background: #f0f0f0;
    }
    .test-case {
      margin: 20px 0;
      background: white;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .test-name {
      font-size: 18px;
      font-weight: bold;
      margin-bottom: 10px;
    }
    .test-description {
      color: #666;
      margin-bottom: 15px;
    }
    .image-container {
      position: relative;
      display: inline-block;
      border: 2px solid #ddd;
    }
    .test-image {
      display: block;
      background: linear-gradient(45deg, #ccc 25%, transparent 25%),
                  linear-gradient(-45deg, #ccc 25%, transparent 25%),
                  linear-gradient(45deg, transparent 75%, #ccc 75%),
                  linear-gradient(-45deg, transparent 75%, #ccc 75%);
      background-size: 20px 20px;
      background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
    }
    .bbox-overlay {
      position: absolute;
      border: 2px dashed #667eea;
      box-sizing: border-box;
    }
    .bbox-label {
      position: absolute;
      top: -25px;
      left: 0;
      background: rgba(0,0,0,0.8);
      color: white;
      padding: 4px 8px;
      font-size: 11px;
      border-radius: 4px;
      font-family: monospace;
    }
    .test-info {
      margin-top: 10px;
      font-size: 14px;
      color: #333;
    }
    .test-info code {
      background: #f5f5f5;
      padding: 2px 6px;
      border-radius: 3px;
      font-family: monospace;
    }
  </style>
</head>
<body>
  <h1>BBox Overlay Visual Tests - Phase 6</h1>
  <p>Verify BBox overlays align correctly across varied aspect ratios.</p>
  <p><strong>Tolerance:</strong> ±2% alignment accuracy</p>
  
  ${testCases.map((testCase, index) => {
    const scaleWidth = Math.min(400, testCase.imageDimensions.width);
    const scaleHeight = (testCase.imageDimensions.height / testCase.imageDimensions.width) * scaleWidth;
    
    return `
    <div class="test-case">
      <div class="test-name">${index + 1}. ${testCase.name}</div>
      <div class="test-description">${testCase.description}</div>
      <div class="image-container">
        <div 
          class="test-image" 
          style="width: ${scaleWidth}px; height: ${scaleHeight}px;"
        ></div>
        <div 
          class="bbox-overlay"
          style="
            left: ${testCase.expectedCSS.left};
            top: ${testCase.expectedCSS.top};
            width: ${testCase.expectedCSS.width};
            height: ${testCase.expectedCSS.height};
          "
        >
          <div class="bbox-label">[${testCase.bbox.join(', ')}]</div>
        </div>
      </div>
      <div class="test-info">
        <div><strong>Image:</strong> <code>${testCase.imageDimensions.width} × ${testCase.imageDimensions.height}</code></div>
        <div><strong>BBox (px):</strong> <code>[x: ${testCase.bbox[0]}, y: ${testCase.bbox[1]}, w: ${testCase.bbox[2]}, h: ${testCase.bbox[3]}]</code></div>
        <div><strong>CSS (%):</strong> <code>left: ${testCase.expectedCSS.left}, top: ${testCase.expectedCSS.top}, width: ${testCase.expectedCSS.width}, height: ${testCase.expectedCSS.height}</code></div>
      </div>
    </div>
    `;
  }).join('\n')}
</body>
</html>
  `;
}

