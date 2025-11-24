# White Screen Troubleshooting Guide

## Quick Diagnosis Steps

### Step 1: Test if React is Working

1. **Go to**: http://localhost:5173/test
2. **Hard refresh**: `Ctrl+Shift+R`

**Expected Results**:
- ✅ **Green "React is Working!" message** → React is fine, SearchDevPage has an issue
- ❌ **Still white screen** → React or dependencies issue

---

### Step 2A: If React Works (Green Message Shows)

The issue is in SearchDevPage. Check browser console:

1. Press `F12` to open DevTools
2. Click "Console" tab
3. Look for red error messages
4. Take screenshot and share the error

**Common Errors**:
- "Cannot find module" → Missing import
- "Unexpected token" → Syntax error
- "X is not defined" → Missing dependency

---

### Step 2B: If Still White Screen

React isn't loading at all. Try these steps:

#### Check 1: Dependencies Installed
```powershell
cd frontend
npm.cmd install
```

#### Check 2: Dev Server Running
Look for terminal showing:
```
VITE v5.4.21  ready in XXXXms
➜  Local:   http://localhost:5173/
```

If not running:
```powershell
cd frontend
npm.cmd run dev
```

#### Check 3: Browser Console
1. Press `F12`
2. Go to "Console" tab
3. Look for ANY errors (red text)
4. Common issues:
   - "Failed to fetch dynamically imported module"
   - "Unexpected token '<'"
   - Network errors

---

## Quick Fixes

### Fix 1: Clear Everything and Restart

```powershell
# Stop frontend (Ctrl+C in terminal)
cd frontend
Remove-Item node_modules -Recurse -Force
Remove-Item package-lock.json -Force
npm.cmd install
npm.cmd run dev
```

### Fix 2: Check Browser Cache

1. Open DevTools (`F12`)
2. Go to "Network" tab
3. Check "Disable cache" checkbox
4. Hard refresh (`Ctrl+Shift+R`)

### Fix 3: Try Different Browser

- Chrome
- Firefox
- Edge

### Fix 4: Check if Port is Blocked

```powershell
netstat -ano | findstr ":5173"
```

Should show:
```
TCP    0.0.0.0:5173    ...    LISTENING    [PID]
```

---

## Browser Console Checks

### Open DevTools
Press `F12` or right-click → "Inspect"

### Check Console Tab
Look for:
- ❌ Red errors
- ⚠️ Yellow warnings
- 🔵 Blue info messages

### Check Network Tab
1. Click "Network" tab
2. Refresh page
3. Look for:
   - Red failed requests (404, 500)
   - Long loading times
   - Blocked requests

### Check Sources Tab
1. Click "Sources" tab
2. Expand `localhost:5173`
3. Look for `/src/main.tsx`
4. Check if files are loading

---

## Common Errors & Solutions

### Error: "Cannot find module 'react-router-dom'"

**Solution**:
```powershell
cd frontend
npm.cmd install react-router-dom
```

### Error: "Unexpected token '<'"

**Cause**: HTML being served instead of JS

**Solution**:
```powershell
# Restart Vite
cd frontend
# Ctrl+C to stop
npm.cmd run dev
```

### Error: "Failed to fetch dynamically imported module"

**Cause**: Build cache issue

**Solution**:
```powershell
cd frontend
Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .vite -Recurse -Force -ErrorAction SilentlyContinue
npm.cmd run dev
```

### Error: Nothing in Console, Just White

**Cause**: CSS hiding content or React not mounting

**Solution**: Open Console and type:
```javascript
document.getElementById('root')
```

Should show: `<div id="root">...</div>`

If it shows `null`, the root div is missing.

---

## What to Share for Help

If still stuck, share:

1. **Browser Console Errors** (screenshot)
2. **Network Tab** (screenshot of failed requests)
3. **Terminal Output** (copy/paste)
4. **Browser & Version** (Chrome 119, Firefox 120, etc.)
5. **URL you're visiting**

---

## Current Test Setup

I created a simple test page at `/test` to diagnose the issue.

**URLs to Try**:
- http://localhost:5173/test (simple test page)
- http://localhost:5173 (redirects to /test)
- http://localhost:5173/dev/search (the actual dev page)

**What You Should See at /test**:
```
✅ React is Working!

If you see this, React is rendering correctly.

Debug Info:
• React: 19.2.0
• Time: [current time]
• URL: http://localhost:5173/test

[Go to Search Dev Page →]
```

---

## Next Steps

1. **Visit**: http://localhost:5173/test
2. **Hard refresh**: `Ctrl+Shift+R`
3. **Report back**:
   - ✅ "I see the green test page!" → Good, issue is in SearchDevPage
   - ❌ "Still white screen" → React isn't loading, share console errors

---

## Manual Check

You can also manually check if files exist:

```powershell
# Check if React components exist
Test-Path frontend\src\pages\SearchDevPage.tsx
Test-Path frontend\src\components\MinScoreSlider.tsx
Test-Path frontend\src\hooks\useUrlState.ts

# Should all return: True
```

---

**Last Updated**: November 14, 2025  
**Current Status**: Added test page at `/test` for diagnosis


