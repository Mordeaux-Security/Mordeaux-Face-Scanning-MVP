# Phase 1-7 Verification Guide

## ✅ How to Verify Your Dev UI is Working

### Step 1: Open the Dev UI
**URL**: http://localhost:5173/dev/search

**What You Should See**:
- React app loads (not the standalone HTML upload page)
- Modern UI with proper styling
- Mock data showing 100 search results

---

### Step 2: Verify Phase 1-7 Features

#### ✅ Phase 0: Dev Search Page Structure
**What to look for**:
- Page title/header
- Query panel at top
- Results area in center
- Controls visible

#### ✅ Phase 4: Non-Functional Shell
**What to look for**:
- Loading states work (if you toggle pageState in console)
- Empty states work
- Error states work
- Components render properly

#### ✅ Phase 5: Image Safety
**What to look for**:
- Images load with SafeImage component
- Broken images show fallback
- No console errors for image loading

#### ✅ Phase 6: Results Rendering
**What to look for**:
- **Query Image Panel**: Top of page, shows sample query image
- **Result Cards (Grid View)**: 
  - Images with thumbnails
  - Score badges (green/yellow/red)
  - Site labels
  - Bounding box overlays on hover
- **Result List Items (List View)**:
  - Detailed metadata
  - Face IDs
  - Timestamps
  - Quality scores

#### ✅ Phase 7: Filters, Pagination & URL Sync
**What to look for**:

**Filters Panel** (top controls):
1. **Min Score Slider**:
   - Slider from 0-100%
   - Current value displayed
   - Gradient visualization
   - Adjusting it filters results in real-time

2. **Site Filter**:
   - Dropdown with "All Sites" option
   - Shows detected sites from results
   - Selecting filters results

3. **Results Counter**:
   - Shows "X of Y results" below filters
   - Updates as you filter

4. **Reset Filters Button**:
   - Clears all filters back to defaults

**View Toggle** (top right):
- Grid view button
- List view button
- Toggle between them works

**Pagination** (bottom):
- Page info: "Showing 1-25 of 100 results"
- First / Previous / Next / Last buttons
- Page numbers (1, 2, 3, 4...)
- Jump to page input
- Page size dropdown (10, 25, 50, 100)

**URL Synchronization**:
- Adjust filters → URL updates automatically
- Example: `/dev/search?minScore=0.75&page=2&pageSize=50`
- Copy URL → Open in new tab → State persists
- Browser back/forward buttons work

---

## 🧪 Quick Tests

### Test 1: Filter by Min Score
1. Move the Min Score slider to 75%
2. Check URL contains `minScore=0.75`
3. Verify fewer results show (only scores ≥ 0.75)
4. Results counter updates

### Test 2: Change Page Size
1. Click page size dropdown
2. Select "50"
3. Check URL contains `pageSize=50`
4. Verify 50 results show per page
5. Page count updates

### Test 3: Navigate Pages
1. Click "Next" button
2. Check URL contains `page=2`
3. Click page number "3"
4. Check URL contains `page=3`
5. Verify different results show

### Test 4: Switch Views
1. Click "List" view button
2. Check URL contains `view=list`
3. Verify layout changes to list view
4. Click "Grid" view button
5. Check URL contains `view=grid`
6. Verify layout changes to grid view

### Test 5: URL Persistence
1. Adjust filters: Min Score 80%, Page 2, Page Size 50
2. Copy URL from browser
3. Open new tab
4. Paste URL
5. Verify all settings persist

### Test 6: Reset Filters
1. Adjust Min Score to 90%
2. Select a site filter
3. Go to page 3
4. Click "Reset Filters" button
5. Verify everything resets to defaults

---

## 🔍 What If It Doesn't Work?

### Problem: Still seeing standalone upload page
**Solution**:
```powershell
# Restart the frontend server
# Press Ctrl+C in the frontend terminal
cd frontend
npm.cmd run dev
```

### Problem: React app loads but shows blank page
**Check**:
1. Open browser DevTools (F12)
2. Check Console tab for errors
3. Look for component render errors
4. Check Network tab for failed requests

### Problem: No filters or pagination visible
**Check**:
1. Make sure you're on `/dev/search` not just `/`
2. Check if SearchDevPage.tsx is loading
3. Look for console errors
4. Verify components are imported correctly

### Problem: URL doesn't update when filtering
**Check**:
1. React Router is installed: `npm list react-router-dom`
2. BrowserRouter is wrapping App
3. useUrlState hook is working
4. Check browser console for errors

---

## 📋 Component Checklist

All these components should be visible and functional:

- [  ] **QueryImage**: Shows sample query image at top
- [  ] **MinScoreSlider**: Slider with gradient, debounced
- [  ] **Site Filter**: Dropdown with site options
- [  ] **Results Counter**: "X of Y results"
- [  ] **Reset Filters**: Button to clear filters
- [  ] **View Toggle**: Grid/List buttons
- [  ] **ResultCard**: Grid view cards with images
- [  ] **ResultListItem**: List view items with details
- [  ] **ScoreBadge**: Color-coded score badges
- [  ] **BBoxOverlay**: Bounding box on hover
- [  ] **SafeImage**: Image with error handling
- [  ] **Pagination**: Full pagination controls
- [  ] **LoadingState**: Loading spinner (when toggled)
- [  ] **EmptyState**: No results message (when filtered to 0)
- [  ] **ErrorState**: Error message (when toggled)

---

## 🎯 Expected Behavior

### Mock Data
- **Total Results**: 100 results
- **Scores**: Range from 0.95 down to 0.15
- **Sites**: 3 different sites (example.com, demo-site.org, test-faces.net)
- **Default View**: Grid view, Page 1, 25 results per page

### Filtering
- Min Score 0 = Shows all 100 results
- Min Score 0.5 = Shows ~50 results
- Min Score 0.75 = Shows ~25 results
- Min Score 0.9 = Shows ~10 results

### Pagination
- Default: Page 1, 25 per page = 4 pages total
- 50 per page = 2 pages total
- 100 per page = 1 page total

### URL State
- Clean URLs (default values not in URL)
- Only non-default values appear in URL
- All state survives page reload
- Browser back/forward works correctly

---

## 🚀 Quick Start Command

If everything seems broken, try a full restart:

```powershell
# Stop frontend (Ctrl+C)
cd frontend
npm.cmd run dev
# Wait for "ready" message
# Open: http://localhost:5173/dev/search
```

---

## ✅ Success Criteria

You know Phase 1-7 is working when:

1. ✅ React app loads (not standalone HTML)
2. ✅ You see 100 mock results
3. ✅ Min Score slider filters results
4. ✅ Site dropdown filters results  
5. ✅ Pagination controls work
6. ✅ Grid/List view toggle works
7. ✅ URL updates when you change settings
8. ✅ Copying URL and opening in new tab preserves state
9. ✅ Reset Filters button clears everything
10. ✅ All components render without errors

---

## 📸 Visual Reference

### What You Should See:

```
┌────────────────────────────────────────────────────────┐
│  SEARCH DEV PAGE                        [Copy URL] [?] │
├────────────────────────────────────────────────────────┤
│                                                         │
│  📸 Query Image                                        │
│  ┌───────────────┐                                    │
│  │               │  Uploaded: 2025-11-14             │
│  │   [Image]     │  Tenant: demo-tenant              │
│  │               │  Backend: Mock Server             │
│  └───────────────┘                                    │
│                                                         │
│  ─────────────────────────────────────────────────    │
│                                                         │
│  Filters                          [Grid] [List]        │
│  ┌──────────────────────────────────────────────┐    │
│  │ Min Score: ████████░░░░░░░░░░ 75%            │    │
│  │ Site: [All Sites ▼]                          │    │
│  │ 25 of 100 results         [Reset Filters]    │    │
│  └──────────────────────────────────────────────┘    │
│                                                         │
│  Results (Grid View)                                   │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                 │
│  │ 95%│ │ 92%│ │ 89%│ │ 86%│ │ 83%│                 │
│  └────┘ └────┘ └────┘ └────┘ └────┘                 │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                 │
│  │ 80%│ │ 77%│ │ 74%│ │ 71%│ │ 68%│                 │
│  └────┘ └────┘ └────┘ └────┘ └────┘                 │
│                                                         │
│  Pagination                                            │
│  Showing 1-25 of 25 results                           │
│  [First] [Prev] [1] 2 3 ... [Next] [Last]            │
│  Page size: [25 ▼]                                    │
│                                                         │
└────────────────────────────────────────────────────────┘
```

---

## 🔗 URLs to Test

- http://localhost:5173/dev/search
- http://localhost:5173/dev/search?minScore=0.75
- http://localhost:5173/dev/search?page=2&pageSize=50
- http://localhost:5173/dev/search?minScore=0.8&site=example.com&page=2&pageSize=50&view=list

---

**Last Updated**: November 14, 2025  
**Status**: React app configured, Phase 1-7 ready for verification


