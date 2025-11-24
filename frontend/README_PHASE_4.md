# Phase 4 - Non-Functional Shell

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

Server starts on: **http://localhost:3000**

### 3. View the Page

Navigate to: **http://localhost:3000/dev/search**

(Root `/` auto-redirects to `/dev/search`)

---

## What's Included

✅ **Routing** - React Router with `/dev/search` route  
✅ **Design Tokens** - Comprehensive design system (`src/tokens.css`)  
✅ **Layout** - Complete skeleton matching Phase 1 wireframes  
✅ **Placeholder Components** - Loading, Empty, Error states  
✅ **Accessibility** - Semantic HTML, ARIA labels, keyboard navigation  
✅ **Responsive** - Mobile, tablet, desktop layouts

---

## Demo Features

On the search page, use the **demo buttons** to toggle between states:

- **Show Loading** - Skeleton loaders with shimmer effect
- **Show Results** - Match grid with 25 placeholder cards
- **Show Empty** - No results state with suggestions
- **Show Error** - Error state with retry actions

---

## File Structure

```
src/
├── components/          # Reusable components
│   ├── LoadingState.*   # Loading placeholder
│   ├── EmptyState.*     # Empty state
│   └── ErrorState.*     # Error state
│
├── pages/               # Page components
│   └── SearchDevPage.*  # Main search page
│
├── App.*                # App shell with routing
├── tokens.css           # Design tokens (MASTER)
└── main.tsx             # Entry point
```

---

## Design Tokens

All design values are centralized in `src/tokens.css`:

```css
/* Colors */
var(--color-primary)
var(--color-success)
var(--color-warning)
var(--color-error)

/* Typography */
var(--font-size-base)
var(--font-weight-semibold)

/* Spacing */
var(--spacing-4)
var(--spacing-6)

/* And more... */
```

---

## NO Business Logic

This is a **visual shell only**:

- ❌ No API calls
- ❌ No data fetching  
- ❌ No real filtering
- ❌ No real pagination

**Coming in Phase 5**: Integration with mock server

---

## Accessibility

- ✅ Semantic HTML (`<header>`, `<main>`, `<section>`)
- ✅ ARIA labels and roles
- ✅ Skip navigation link
- ✅ Keyboard accessible
- ✅ Focus indicators
- ✅ Screen reader friendly

Test with:
- Tab key for keyboard navigation
- Screen reader (NVDA/JAWS/VoiceOver)
- Lighthouse audit in DevTools

---

## Browser Support

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)

---

## Documentation

Full documentation: `../docs/PHASE_4_NON_FUNCTIONAL_SHELL_COMPLETE.md`

---

**Phase**: 4 - Non-Functional Shell  
**Status**: ✅ Complete  
**Next**: Phase 5 - Mock Server Integration

