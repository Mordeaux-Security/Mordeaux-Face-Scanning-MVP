# UAT Script - Phases 1-14
## Mordeaux Face Scanning MVP - Frontend

**Version:** 1.0  
**Date:** November 15, 2025  
**Test Environment:** Localhost Development

---

## Overview

This UAT (User Acceptance Testing) script covers all implemented phases (1-14) of the frontend development. Each test case includes preconditions, steps, expected results, and pass/fail criteria.

---

## Preconditions

### Environment Setup
```bash
# 1. Install dependencies
cd frontend
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env as needed

# 3. Start development server
npm run dev
```

### Test Data
- Mock server running on `http://localhost:8000`
- Sample images available in `samples/` directory
- At least 100 mock results configured

---

## Test Cases

### TC-001: Dev Mode Access Control
**Priority:** HIGH  
**Phase:** 10

**Preconditions:**
- Dev mode enabled in `.env`

**Steps:**
1. Navigate to `http://localhost:5173/dev/search`
2. Verify page loads without redirect
3. Open console and check for `[DevRouteGuard]` logs

**Expected Results:**
- ✅ Page loads successfully
- ✅ No unauthorized redirect
- ✅ Console shows access granted

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-002: Basic Search Interface
**Priority:** HIGH  
**Phase:** 1-6

**Steps:**
1. Navigate to `/dev/search`
2. Verify query panel visible
3. Verify results section visible
4. Verify all UI elements present

**Expected Results:**
- ✅ Query image placeholder shown
- ✅ Filters panel visible
- ✅ Results grid/list toggle visible
- ✅ No console errors

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-003: Grid View Rendering
**Priority:** HIGH  
**Phase:** 6

**Steps:**
1. Ensure "Grid" view is selected
2. Click "Show Results" button (demo controls)
3. Verify results render in grid layout
4. Count visible result cards

**Expected Results:**
- ✅ Results display in grid (3-4 columns)
- ✅ Each card shows thumbnail, score, metadata
- ✅ Hover shows bounding box overlay
- ✅ No broken images

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-004: List View Rendering
**Priority:** HIGH  
**Phase:** 6

**Steps:**
1. Click "List" view button
2. Verify results re-render in list layout
3. Verify metadata visible in rows

**Expected Results:**
- ✅ Results display in list format
- ✅ Each row shows thumbnail + metadata
- ✅ Metadata includes face_id, timestamp, quality
- ✅ Smooth transition from grid

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-005: Min Score Filter
**Priority:** HIGH  
**Phase:** 7

**Steps:**
1. Note current result count
2. Drag min score slider to 0.8
3. Wait for debounce (300ms)
4. Verify results filtered

**Expected Results:**
- ✅ Slider updates visually
- ✅ Results filtered to score ≥ 0.8
- ✅ Result count updates
- ✅ URL updates with `?minScore=0.8`

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-006: Site Filter
**Priority:** MEDIUM  
**Phase:** 7

**Steps:**
1. Select "example.com" from site filter
2. Verify results filtered
3. Check result count

**Expected Results:**
- ✅ Only results from "example.com" shown
- ✅ Result count updates
- ✅ URL updates with `?site=example.com`

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-007: Pagination
**Priority:** HIGH  
**Phase:** 7

**Steps:**
1. Verify pagination controls visible
2. Note current page (1)
3. Click "Next" button
4. Verify page 2 loads

**Expected Results:**
- ✅ Page number updates to 2
- ✅ Different results shown
- ✅ URL updates with `?page=2`
- ✅ "Previous" button now enabled

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-008: URL State Sync
**Priority:** HIGH  
**Phase:** 7

**Steps:**
1. Set filters: minScore=0.75, site=demo-site.org, page=2
2. Copy URL from address bar
3. Open new tab
4. Paste URL and navigate

**Expected Results:**
- ✅ All filters restored (minScore, site, page)
- ✅ Correct results shown
- ✅ UI state matches URL

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-009: Browser Back/Forward
**Priority:** MEDIUM  
**Phase:** 7

**Steps:**
1. Change minScore to 0.6
2. Change to page 2
3. Click browser back button twice
4. Click forward button once

**Expected Results:**
- ✅ State reverts correctly
- ✅ Results update on each navigation
- ✅ No page reload required

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-010: Safe External Links
**Priority:** HIGH  
**Phase:** 8

**Steps:**
1. Find a result card
2. Click "🔗 Source" link
3. Verify link opens in new tab
4. Inspect link attributes (dev tools)

**Expected Results:**
- ✅ Link opens in new tab
- ✅ Has `rel="noreferrer noopener nofollow"`
- ✅ No javascript: or data: URLs allowed
- ✅ Console shows `[SafeLink] LINK_ALLOWED`

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-011: Storage Chip Display
**Priority:** MEDIUM  
**Phase:** 8

**Steps:**
1. Examine result cards
2. Verify storage chip visible (MinIO/S3/External)
3. Hover over chip for tooltip

**Expected Results:**
- ✅ Storage chip shown on each card
- ✅ Correct provider detected (MinIO for mock)
- ✅ Icon and label visible

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-012: Virtualization Toggle
**Priority:** HIGH  
**Phase:** 9

**Steps:**
1. Check "Use Virtualization" toggle
2. Verify indicator message appears
3. Scroll through results
4. Measure scroll performance

**Expected Results:**
- ✅ Message: "⚡ Virtualized mode: rendering X results"
- ✅ Smooth scrolling (60fps)
- ✅ Pagination hidden
- ✅ All results in viewport

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-013: Performance Metrics
**Priority:** MEDIUM  
**Phase:** 9

**Steps:**
1. Open browser console
2. Enable virtualization
3. Scroll to bottom of results
4. Check console for performance logs

**Expected Results:**
- ✅ `[Performance]` logs visible
- ✅ Render times < 16ms
- ✅ Filter duration < 50ms
- ✅ No memory leaks

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-014: Data Redaction Toggle
**Priority:** MEDIUM  
**Phase:** 10

**Steps:**
1. Locate "🔓 Reveal Sensitive Data" toggle (dev mode)
2. Verify initial state (redacted)
3. Check toggle
4. Verify data revealed

**Expected Results:**
- ✅ Toggle only visible in dev mode
- ✅ Data initially redacted
- ✅ Toggle reveals full data
- ✅ Console logs toggle state

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-015: Debug Panel
**Priority:** MEDIUM  
**Phase:** 11

**Steps:**
1. Click "Debug Panel" button (bottom-right)
2. Verify panel expands
3. Switch between tabs (Metrics, Events, Logs)
4. Click "Export Logs"

**Expected Results:**
- ✅ Panel opens smoothly
- ✅ Metrics tab shows FPS, memory
- ✅ Events tab shows counters
- ✅ Logs tab shows recent logs
- ✅ Export downloads JSON file

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-016: Keyboard Navigation
**Priority:** HIGH  
**Phase:** 12

**Steps:**
1. Use Tab key to navigate through UI
2. Verify focus indicators visible
3. Press Enter/Space on buttons
4. Test Escape key (if modals exist)

**Expected Results:**
- ✅ Focus visible on all interactive elements
- ✅ Tab order logical
- ✅ Enter/Space activates buttons
- ✅ Skip link appears on Tab

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-017: Screen Reader Compatibility
**Priority:** MEDIUM  
**Phase:** 12

**Preconditions:**
- Screen reader installed (NVDA/JAWS)

**Steps:**
1. Start screen reader
2. Navigate through page
3. Verify announcements
4. Test form controls

**Expected Results:**
- ✅ All images have alt text
- ✅ Buttons have labels
- ✅ Form controls announced
- ✅ Live regions work

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-018: Touch Target Sizing
**Priority:** MEDIUM  
**Phase:** 12

**Steps:**
1. Use browser dev tools
2. Measure interactive elements
3. Verify minimum 40x40px

**Expected Results:**
- ✅ All buttons ≥ 40x40px
- ✅ Adequate spacing between elements
- ✅ Touch-friendly on mobile

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-019: Responsive Breakpoints
**Priority:** MEDIUM  
**Phase:** 12

**Steps:**
1. Resize browser to mobile (< 640px)
2. Resize to tablet (640-1024px)
3. Resize to desktop (> 1024px)
4. Verify layout adapts

**Expected Results:**
- ✅ Mobile: 1 column grid
- ✅ Tablet: 2 column grid
- ✅ Desktop: 3-4 column grid
- ✅ No horizontal scroll

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

### TC-020: API Integration (Mock)
**Priority:** HIGH  
**Phase:** 13

**Steps:**
1. Verify `.env` has `VITE_USE_REAL_API=false`
2. Make a search request
3. Check network tab
4. Verify mock server called

**Expected Results:**
- ✅ Request goes to `http://localhost:8000`
- ✅ Mock data returned
- ✅ Results display correctly
- ✅ Console shows API logs

**Pass/Fail:**
- [ ] PASS
- [ ] FAIL (Details: _______)

---

## Summary Checklist

### Phase 1-8 (Foundation)
- [ ] TC-001: Dev mode access control
- [ ] TC-002: Basic search interface
- [ ] TC-003: Grid view rendering
- [ ] TC-004: List view rendering
- [ ] TC-005: Min score filter
- [ ] TC-006: Site filter
- [ ] TC-007: Pagination
- [ ] TC-008: URL state sync
- [ ] TC-009: Browser back/forward
- [ ] TC-010: Safe external links
- [ ] TC-011: Storage chip display

### Phase 9-11 (Advanced Features)
- [ ] TC-012: Virtualization toggle
- [ ] TC-013: Performance metrics
- [ ] TC-014: Data redaction toggle
- [ ] TC-015: Debug panel

### Phase 12 (Accessibility)
- [ ] TC-016: Keyboard navigation
- [ ] TC-017: Screen reader compatibility
- [ ] TC-018: Touch target sizing
- [ ] TC-019: Responsive breakpoints

### Phase 13 (Backend)
- [ ] TC-020: API integration (mock)

---

## Edge Cases

### EC-001: Empty Results
**Steps:** Filter to yield 0 results  
**Expected:** Empty state component shown

### EC-002: Network Error
**Steps:** Stop mock server, trigger search  
**Expected:** Error state component shown

### EC-003: Large Result Set (2000+)
**Steps:** Generate 2000 results, enable virtualization  
**Expected:** Smooth scrolling, no lag

### EC-004: Rapid Filter Changes
**Steps:** Drag slider rapidly back and forth  
**Expected:** Debouncing works, no URL spam

### EC-005: Invalid URL Params
**Steps:** Navigate to `?minScore=invalid&page=-1`  
**Expected:** Graceful fallback to defaults

---

## Performance Benchmarks

| Metric | Target | Actual | Pass/Fail |
|--------|--------|--------|-----------|
| First Interactive | < 2s | _____ | [ ] |
| Scroll FPS (2k results) | 60fps | _____ | [ ] |
| Filter Update | < 50ms | _____ | [ ] |
| Bundle Size | < 500KB | _____ | [ ] |
| Memory Usage | Stable | _____ | [ ] |

---

## Test Environment

**Browser:** _____________________  
**OS:** _____________________  
**Screen Resolution:** _____________________  
**Network Speed:** _____________________

---

## Sign-Off

**Tester Name:** _____________________  
**Date:** _____________________  
**Overall Result:** [ ] PASS / [ ] FAIL  
**Notes:**

---



