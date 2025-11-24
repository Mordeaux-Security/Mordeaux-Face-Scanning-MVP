# Phase 6 — Results Rendering Quick Reference

## 🚀 Quick Start

```bash
cd frontend
npm run dev
```

Navigate to: `http://localhost:5173/dev/search`

## 🎯 What's New in Phase 6

### New Components

1. **BBoxOverlay** - Bounding box visualization
2. **ScoreBadge** - Color-coded similarity scores
3. **DistanceChip** - Distance metrics display
4. **ResultCard** - Grid view result card
5. **ResultListItem** - List view result item

### Features

- ✅ Grid/List view toggle
- ✅ BBox overlay on hover
- ✅ Color-coded score badges
- ✅ Distance metrics (optional)
- ✅ Responsive layouts
- ✅ Full accessibility support

## 🎨 View Modes

### Grid View
- Fast visual scanning
- 5/3/1 columns (desktop/tablet/mobile)
- Compact metadata display

### List View
- Detailed metadata
- Vertical layout
- Extended information display

## 🧪 Testing BBox Alignment

The BBox overlay converts pixel coordinates to CSS percentages with ±2% accuracy.

### Test Coverage
- ✅ 12 aspect ratios tested
- ✅ Edge cases covered
- ✅ Visual tests available

### Run Visual Tests

Generate HTML test file:
```bash
# Create visual test HTML from test suite
node -e "const { generateVisualTestHTML, BBOX_TEST_CASES } = require('./src/components/BBoxOverlay.test.tsx'); console.log(generateVisualTestHTML(BBOX_TEST_CASES));" > bbox-tests.html
```

Open `bbox-tests.html` in browser to visually inspect BBox alignment.

## 📊 Score Badge Color Coding

| Score | Color | Meaning |
|-------|-------|---------|
| ≥ 80% | 🟢 Green | High confidence |
| 60-79% | 🟡 Yellow | Medium confidence |
| < 60% | 🔴 Red | Low confidence |

## 🔧 Component Usage

### BBoxOverlay

```tsx
import BBoxOverlay, { useImageDimensions } from './components/BBoxOverlay';

function MyComponent() {
  const [imageDimensions, imgRef] = useImageDimensions();
  
  return (
    <div style={{ position: 'relative' }}>
      <img ref={imgRef} src={imageUrl} alt="Face" />
      <BBoxOverlay
        bbox={[100, 100, 200, 200]} // [x, y, width, height]
        imageDimensions={imageDimensions}
        showCoordinates={true}
      />
    </div>
  );
}
```

### ScoreBadge

```tsx
import ScoreBadge from './components/ScoreBadge';

<ScoreBadge 
  score={0.85}
  format="percentage"
  size="medium"
  showIcon={true}
/>
```

### ResultCard (Grid)

```tsx
import ResultCard from './components/ResultCard';

<div className="match-grid">
  {results.map(hit => (
    <ResultCard 
      key={hit.face_id} 
      hit={hit}
      showDistance={false}
      onCopyId={(id) => console.log('Copied:', id)}
    />
  ))}
</div>
```

### ResultListItem (List)

```tsx
import ResultListItem from './components/ResultListItem';

<div className="match-list">
  {results.map(hit => (
    <ResultListItem 
      key={hit.face_id} 
      hit={hit}
      showDistance={true}
      onCopyId={(id) => console.log('Copied:', id)}
    />
  ))}
</div>
```

## 📁 New Files

```
frontend/src/
├── components/
│   ├── BBoxOverlay.tsx
│   ├── BBoxOverlay.css
│   ├── BBoxOverlay.test.tsx
│   ├── ScoreBadge.tsx
│   ├── ScoreBadge.css
│   ├── DistanceChip.tsx
│   ├── DistanceChip.css
│   ├── ResultCard.tsx
│   ├── ResultCard.css
│   ├── ResultListItem.tsx
│   └── ResultListItem.css
└── pages/
    ├── SearchDevPage.tsx (updated)
    └── SearchDevPage.css (updated)
```

## 🎯 Key Features

### BBox Overlay
- Converts pixel coordinates to CSS percentages
- ±2% accuracy tolerance
- Shows on hover
- Optional coordinates display
- Validates bbox bounds

### Score Badges
- Three color variants
- Three size options
- Optional icon
- Accessible labels

### Distance Chip
- Cosine/Euclidean/Manhattan support
- Monospace font
- Tabular number formatting

### Result Cards
- Safe image loading
- Hover effects
- Responsive design
- Accessibility compliant

## 🔍 Mock Data

The SearchDevPage uses mock data for testing:
- 25 results with realistic scores (0.95 - 0.70)
- Random bbox coordinates
- Varied timestamps
- Multiple sites (example.com, demo-site.org, test-faces.net)
- Quality scores and p-hashes

## 📱 Responsive Design

- **Desktop (≥1024px):** 5-column grid
- **Tablet (768-1023px):** 3-column grid
- **Mobile (<768px):** 1-column grid/list

## ⌨️ Keyboard Navigation

- Tab through results
- Enter to activate actions
- Escape to close modals (future phase)
- Focus visible indicators

## 🎨 Design Tokens Used

- `--color-success` (green badges)
- `--color-warning` (yellow badges)
- `--color-error` (red badges)
- `--color-primary` (hover states)
- `--shadow-sm/md` (card elevations)
- `--radius-lg` (card borders)
- `--spacing-*` (consistent spacing)
- `--z-dropdown/tooltip` (overlays)

## 🐛 Troubleshooting

### BBox Not Showing
- Check if image has loaded (dimensions > 0)
- Verify bbox coordinates are within image bounds
- Ensure container has `position: relative`

### Score Badge Wrong Color
- Verify score is between 0-1 (not 0-100)
- Check color token values in `tokens.css`

### Grid Layout Broken
- Check container width
- Verify CSS variables are loaded
- Test different screen sizes

## 📚 Documentation

- [Phase 6 Complete](../docs/PHASE_6_RESULTS_RENDERING_COMPLETE.md)
- [BBox Test Cases](./src/components/BBoxOverlay.test.tsx)

---

**Phase 6 Status:** ✅ Complete

