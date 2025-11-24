# Phase 7 — Filters, Pagination & URL Sync Quick Reference

## 🚀 Quick Start

```bash
cd frontend
npm run dev
```

Navigate to: `http://localhost:5173/dev/search`

## 🎯 What's New in Phase 7

### New Components

1. **MinScoreSlider** - Visual filter slider with debouncing
2. **Pagination** - Full-featured pagination with page size control
3. **useUrlState Hook** - URL state synchronization

### New Features

- ✅ Min score filtering (0-100%)
- ✅ Site filtering (dropdown)
- ✅ Pagination with First/Prev/Next/Last
- ✅ Page size selector (10/25/50/100)
- ✅ URL state synchronization (deep-linking)
- ✅ State persists across reloads
- ✅ Copy URL button
- ✅ Reset filters button

## 🔗 URL Parameters

All state is synced to URL parameters:

| Parameter | Values | Default |
|-----------|--------|---------|
| `minScore` | 0-1 | 0 |
| `site` | domain string | '' (all) |
| `page` | ≥1 | 1 |
| `pageSize` | 10, 25, 50, 100 | 25 |
| `view` | 'grid', 'list' | 'grid' |

### Example URLs

```
# Default
/dev/search

# With filters
/dev/search?minScore=0.75&site=example.com

# Full state
/dev/search?minScore=0.8&site=demo-site.org&page=2&pageSize=50&view=list
```

## 📋 Testing

Run the QA test script to verify state ↔ URL round-trip:

**File:** `docs/QA_SCRIPT_PHASE_7.md`

**Quick Test:**
1. Adjust Min Score slider to 75%
2. Check URL contains `minScore=0.75`
3. Copy URL
4. Open in new tab
5. Verify slider shows 75%

## 🎨 New UI Elements

### Filters Panel
- Min Score Slider with gradient visualization
- Site dropdown filter
- Results counter ("X of Y results")
- Reset Filters button

### Pagination
- Page info ("Showing 1-25 of 100")
- First/Previous/Next/Last buttons
- Page number buttons with ellipsis
- Jump to page input
- Page size selector

### Header
- Copy URL button (shares current state)

## 🔧 Component Usage

### useUrlState Hook

```tsx
import { useUrlState, urlParsers } from './hooks/useUrlState';

const [state, setState, resetState] = useUrlState({
  minScore: { default: 0, parse: urlParsers.number },
  page: { default: 1, parse: urlParsers.int },
});

// Update state (syncs to URL)
setState({ minScore: 0.75, page: 1 });

// Reset to defaults
resetState();
```

### MinScoreSlider

```tsx
import MinScoreSlider from './components/MinScoreSlider';

<MinScoreSlider
  value={0.75}
  onChange={(value) => console.log(value)}
  debounceMs={300}
  showLabel={true}
/>
```

### Pagination

```tsx
import Pagination from './components/Pagination';

<Pagination
  currentPage={2}
  totalPages={10}
  totalItems={250}
  itemsPerPage={25}
  onPageChange={(page) => console.log(page)}
  onPageSizeChange={(size) => console.log(size)}
  pageSizeOptions={[10, 25, 50, 100]}
/>
```

## 📁 New Files

```
frontend/src/
├── components/
│   ├── MinScoreSlider.tsx
│   ├── MinScoreSlider.css
│   ├── Pagination.tsx
│   └── Pagination.css
├── hooks/
│   └── useUrlState.ts
└── pages/
    ├── SearchDevPage.tsx (updated)
    ├── SearchDevPage.css
    └── SearchDevPage_Phase7.css

docs/
├── QA_SCRIPT_PHASE_7.md
└── PHASE_7_FILTERS_PAGINATION_URL_SYNC_COMPLETE.md
```

## 🎯 Key Features

### URL State Sync
- Automatic bidirectional sync
- Works with browser back/forward
- Persists across reloads
- Copy/paste URLs to share state

### Debouncing
- Slider updates debounced (300ms)
- Prevents URL spam
- Smooth UX

### Smart Defaults
- Clean URLs (defaults not included)
- Invalid params fallback gracefully
- Page resets on filter change

## ⌨️ Keyboard Shortcuts

- **Tab**: Navigate through controls
- **Space**: Toggle view mode buttons
- **Arrow Keys**: Navigate pagination
- **Enter**: Jump to page

## 🐛 Troubleshooting

### Filters Not Working
- Check browser console for errors
- Verify useUrlState hook is imported
- Ensure URL params are being set

### URL Not Updating
- Check React Router is set up correctly
- Verify useSearchParams is available
- Check for JavaScript errors

### State Not Persisting
- Verify URL contains parameters
- Check browser history
- Test in different browsers

## 📚 Documentation

- [Phase 7 Complete](../docs/PHASE_7_FILTERS_PAGINATION_URL_SYNC_COMPLETE.md)
- [QA Test Script](../docs/QA_SCRIPT_PHASE_7.md)
- [useUrlState Hook](./src/hooks/useUrlState.ts)

---

**Phase 7 Status:** ✅ Complete

All state now syncs to URL. Deep-linking works perfectly!





