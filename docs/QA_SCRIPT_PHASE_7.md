# Phase 7 QA Test Script — URL State Round-Trip

## Overview

This script verifies that all application state correctly syncs with URL parameters,
enabling deep-linking and state persistence across page reloads.

**Goal:** Confirm `state ↔ URL` round-trip works reliably.

---

## Test Environment

- **URL:** `http://localhost:5173/dev/search`
- **Browser:** Chrome, Firefox, or Safari (test in all three)
- **Requirements:** Frontend dev server running (`npm run dev`)

---

## Test Cases

### ✅ Test 1: Min Score Filter → URL

**Steps:**
1. Navigate to `/dev/search`
2. Adjust Min Score slider to `75%`
3. Check URL

**Expected:**
- URL should update to: `/dev/search?minScore=0.75`
- Results should filter to only show scores ≥ 75%

**Pass Criteria:**
- [ ] URL contains `minScore=0.75`
- [ ] Results update in real-time
- [ ] No page reload occurs

---

### ✅ Test 2: Site Filter → URL

**Steps:**
1. Navigate to `/dev/search`
2. Select "example.com" from Site filter
3. Check URL

**Expected:**
- URL should update to: `/dev/search?site=example.com`
- Results should only show hits from example.com

**Pass Criteria:**
- [ ] URL contains `site=example.com`
- [ ] Only example.com results visible
- [ ] Filter count updates correctly

---

### ✅ Test 3: Pagination → URL

**Steps:**
1. Navigate to `/dev/search`
2. Click "Next" or page 2
3. Check URL

**Expected:**
- URL should update to: `/dev/search?page=2`
- Results should show page 2

**Pass Criteria:**
- [ ] URL contains `page=2`
- [ ] Different results shown
- [ ] Page indicator shows "2"

---

### ✅ Test 4: Page Size → URL

**Steps:**
1. Navigate to `/dev/search`
2. Change page size to 50
3. Check URL

**Expected:**
- URL should update to: `/dev/search?pageSize=50`
- More results visible per page

**Pass Criteria:**
- [ ] URL contains `pageSize=50`
- [ ] 50 results shown (or fewer if limited)
- [ ] Pagination updates to reflect new size

---

### ✅ Test 5: View Mode → URL

**Steps:**
1. Navigate to `/dev/search`
2. Click "List" view
3. Check URL

**Expected:**
- URL should update to: `/dev/search?view=list`
- Layout switches to list view

**Pass Criteria:**
- [ ] URL contains `view=list`
- [ ] List view displayed
- [ ] Toggle button shows "List" as active

---

### ✅ Test 6: Combined Filters → URL

**Steps:**
1. Navigate to `/dev/search`
2. Set Min Score to 60%
3. Select "demo-site.org" from Site filter
4. Set page size to 10
5. Navigate to page 2
6. Switch to List view
7. Check URL

**Expected:**
- URL should contain ALL parameters:
  ```
  /dev/search?minScore=0.6&site=demo-site.org&pageSize=10&page=2&view=list
  ```

**Pass Criteria:**
- [ ] URL contains all 5 parameters
- [ ] Results reflect all filters
- [ ] List view active
- [ ] Page 2 displayed

---

### ✅ Test 7: URL → State (Deep Link)

**Steps:**
1. Close browser tab
2. Open new tab
3. Paste URL: `http://localhost:5173/dev/search?minScore=0.8&site=example.com&page=3&pageSize=25&view=list`
4. Press Enter

**Expected:**
- Page loads with:
  - Min Score: 80%
  - Site: example.com
  - Page: 3
  - Page Size: 25
  - View: List

**Pass Criteria:**
- [ ] Min Score slider shows 80%
- [ ] Site dropdown shows "example.com"
- [ ] Page 3 is active
- [ ] Page size is 25
- [ ] List view is displayed
- [ ] Correct filtered results shown

---

### ✅ Test 8: Page Reload Persistence

**Steps:**
1. Navigate to `/dev/search`
2. Set Min Score to 70%
3. Select "test-faces.net"
4. Go to page 2
5. Press F5 (reload page)

**Expected:**
- After reload:
  - Min Score: 70%
  - Site: test-faces.net
  - Page: 2
  - All state preserved

**Pass Criteria:**
- [ ] URL unchanged after reload
- [ ] All filters persist
- [ ] Same page displayed
- [ ] Results unchanged

---

### ✅ Test 9: Browser Back/Forward

**Steps:**
1. Navigate to `/dev/search`
2. Set Min Score to 50%
3. Click Next page (page 2)
4. Click Next page again (page 3)
5. Click browser Back button
6. Click browser Back button again
7. Click browser Forward button

**Expected:**
- Back to page 2 → URL: `?minScore=0.5&page=2`
- Back to page 1 → URL: `?minScore=0.5`
- Forward to page 2 → URL: `?minScore=0.5&page=2`

**Pass Criteria:**
- [ ] Back button navigates through state history
- [ ] Forward button works correctly
- [ ] URL updates with each navigation
- [ ] Results update accordingly

---

### ✅ Test 10: Copy URL & Share

**Steps:**
1. Navigate to `/dev/search`
2. Set filters:
   - Min Score: 85%
   - Site: example.com
   - Page 2
   - List view
3. Click "📋 Copy URL" button
4. Open incognito/private window
5. Paste URL and navigate

**Expected:**
- Incognito window shows EXACT same state:
  - Min Score: 85%
  - Site: example.com
  - Page: 2
  - List view

**Pass Criteria:**
- [ ] Copy URL button works
- [ ] Pasted URL restores exact state
- [ ] No data loss across browsers

---

### ✅ Test 11: Reset Filters

**Steps:**
1. Navigate to `/dev/search`
2. Set filters:
   - Min Score: 75%
   - Site: demo-site.org
3. Click "Reset Filters" button
4. Check URL

**Expected:**
- URL returns to: `/dev/search` (no parameters)
- All filters reset to defaults:
  - Min Score: 0%
  - Site: All Sites
  - Page: 1
  - Page Size: 25
  - View: Grid

**Pass Criteria:**
- [ ] URL cleared of parameters
- [ ] All filters reset
- [ ] Default state restored
- [ ] All results visible

---

### ✅ Test 12: Invalid URL Parameters

**Steps:**
1. Navigate to: `http://localhost:5173/dev/search?minScore=abc&page=-5&view=invalid`

**Expected:**
- App handles invalid values gracefully:
  - `minScore=abc` → defaults to 0
  - `page=-5` → defaults to 1
  - `view=invalid` → defaults to 'grid'

**Pass Criteria:**
- [ ] No errors or crashes
- [ ] Invalid params replaced with defaults
- [ ] URL may be corrected on first interaction
- [ ] Console warning logged (check DevTools)

---

### ✅ Test 13: Edge Case — Empty Results

**Steps:**
1. Navigate to `/dev/search`
2. Set Min Score to 100%

**Expected:**
- URL: `/dev/search?minScore=1`
- "No results" message displayed
- Pagination hidden or shows 0 results

**Pass Criteria:**
- [ ] Empty state component shown
- [ ] No crash or error
- [ ] URL still contains filter
- [ ] Can reset filters to see results again

---

### ✅ Test 14: Rapid Filter Changes

**Steps:**
1. Navigate to `/dev/search`
2. Rapidly move Min Score slider back and forth for 10 seconds
3. Stop and wait 1 second
4. Check URL

**Expected:**
- URL stabilizes to final slider position
- Debouncing prevents excessive URL updates
- No performance issues

**Pass Criteria:**
- [ ] URL updates after debounce delay (~300ms)
- [ ] No browser history spam
- [ ] Slider remains responsive
- [ ] Final state is correct

---

### ✅ Test 15: Multiple Tabs Sync

**Steps:**
1. Open `/dev/search` in Tab 1
2. Duplicate tab (Tab 2)
3. In Tab 1: Set Min Score to 80%
4. In Tab 2: Refresh page

**Expected:**
- Tab 1: URL shows `?minScore=0.8`
- Tab 2: Shows default state (tabs don't sync automatically)

**Pass Criteria:**
- [ ] Tabs are independent
- [ ] Each tab has its own URL state
- [ ] No unexpected syncing
- [ ] Both tabs functional

---

## Summary Checklist

### State → URL
- [ ] Min Score filter updates URL
- [ ] Site filter updates URL
- [ ] Pagination updates URL
- [ ] Page size updates URL
- [ ] View mode updates URL
- [ ] Combined filters work correctly

### URL → State
- [ ] Deep links restore full state
- [ ] Page reload preserves state
- [ ] Browser back/forward work
- [ ] Copy/paste URL works
- [ ] Invalid params handled gracefully

### User Experience
- [ ] No unnecessary page reloads
- [ ] Debouncing prevents URL spam
- [ ] Reset filters clears URL
- [ ] Empty results handled gracefully
- [ ] Performance acceptable

---

## Browser Compatibility

Test the above in:
- [ ] Chrome/Chromium
- [ ] Firefox
- [ ] Safari
- [ ] Edge

---

## Debugging Tips

If a test fails:

1. **Check Console:** Look for errors in DevTools
2. **Inspect URL:** Manually verify search params
3. **Check React DevTools:** Verify component state
4. **Network Tab:** Ensure no unexpected requests
5. **Check `useUrlState` Hook:** Add console.logs if needed

---

## Success Criteria

**ALL 15 tests must pass** for Phase 7 completion.

If any test fails:
1. Document the failure
2. Fix the issue
3. Re-run all tests
4. Confirm fix doesn't break other tests

---

## Notes

- **URL Parameters Used:**
  - `minScore` (number, 0-1)
  - `site` (string)
  - `page` (integer, ≥1)
  - `pageSize` (integer)
  - `view` ('grid' | 'list')

- **Default Values:**
  - `minScore`: 0
  - `site`: '' (empty, all sites)
  - `page`: 1
  - `pageSize`: 25
  - `view`: 'grid'

- **Debounce Delay:** 300ms for Min Score slider

---

**Last Updated:** November 14, 2025  
**Phase:** 7 — Filters, Pagination, and URL Sync





