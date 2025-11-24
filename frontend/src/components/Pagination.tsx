/**
 * Pagination Component - Phase 7
 * ===============================
 * 
 * Pagination controls with page navigation.
 * 
 * Features:
 * - Shows total and current page
 * - Previous/Next navigation
 * - Jump to specific page
 * - Page size selector
 * - Keyboard accessible
 * - Responsive design
 */

import './Pagination.css';

interface PaginationProps {
  currentPage: number;        // 1-indexed
  totalPages: number;
  totalItems: number;
  itemsPerPage: number;
  onPageChange: (page: number) => void;
  onPageSizeChange?: (pageSize: number) => void;
  pageSizeOptions?: number[];
  disabled?: boolean;
}

export default function Pagination({
  currentPage,
  totalPages,
  totalItems,
  itemsPerPage,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = [10, 25, 50, 100],
  disabled = false,
}: PaginationProps) {
  const startItem = (currentPage - 1) * itemsPerPage + 1;
  const endItem = Math.min(currentPage * itemsPerPage, totalItems);
  
  const handlePrevious = () => {
    if (currentPage > 1) {
      onPageChange(currentPage - 1);
    }
  };
  
  const handleNext = () => {
    if (currentPage < totalPages) {
      onPageChange(currentPage + 1);
    }
  };
  
  const handleFirst = () => {
    if (currentPage !== 1) {
      onPageChange(1);
    }
  };
  
  const handleLast = () => {
    if (currentPage !== totalPages) {
      onPageChange(totalPages);
    }
  };
  
  const handlePageInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const page = parseInt(e.target.value, 10);
    if (page >= 1 && page <= totalPages) {
      onPageChange(page);
    }
  };
  
  const handlePageSizeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newPageSize = parseInt(e.target.value, 10);
    onPageSizeChange?.(newPageSize);
  };
  
  // Generate visible page numbers
  const getVisiblePages = (): (number | 'ellipsis')[] => {
    const pages: (number | 'ellipsis')[] = [];
    const maxVisible = 7;
    
    if (totalPages <= maxVisible) {
      // Show all pages
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      // Show first, last, and pages around current
      pages.push(1);
      
      if (currentPage > 3) {
        pages.push('ellipsis');
      }
      
      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);
      
      for (let i = start; i <= end; i++) {
        pages.push(i);
      }
      
      if (currentPage < totalPages - 2) {
        pages.push('ellipsis');
      }
      
      pages.push(totalPages);
    }
    
    return pages;
  };
  
  const visiblePages = getVisiblePages();
  
  if (totalItems === 0) {
    return (
      <div className="pagination empty">
        <span className="pagination-info">No results</span>
      </div>
    );
  }
  
  return (
    <div className={`pagination ${disabled ? 'disabled' : ''}`}>
      {/* Info */}
      <div className="pagination-info">
        Showing <strong>{startItem}</strong> to <strong>{endItem}</strong> of{' '}
        <strong>{totalItems}</strong> results
      </div>
      
      {/* Controls */}
      <div className="pagination-controls">
        {/* First */}
        <button
          className="pagination-button"
          onClick={handleFirst}
          disabled={disabled || currentPage === 1}
          aria-label="First page"
          title="First page"
        >
          ⏮
        </button>
        
        {/* Previous */}
        <button
          className="pagination-button"
          onClick={handlePrevious}
          disabled={disabled || currentPage === 1}
          aria-label="Previous page"
          title="Previous page"
        >
          ◀
        </button>
        
        {/* Page Numbers */}
        <div className="pagination-pages" role="navigation" aria-label="Pagination">
          {visiblePages.map((page, index) => {
            if (page === 'ellipsis') {
              return (
                <span key={`ellipsis-${index}`} className="pagination-ellipsis">
                  …
                </span>
              );
            }
            
            return (
              <button
                key={page}
                className={`pagination-page ${currentPage === page ? 'active' : ''}`}
                onClick={() => onPageChange(page)}
                disabled={disabled}
                aria-label={`Page ${page}`}
                aria-current={currentPage === page ? 'page' : undefined}
              >
                {page}
              </button>
            );
          })}
        </div>
        
        {/* Next */}
        <button
          className="pagination-button"
          onClick={handleNext}
          disabled={disabled || currentPage === totalPages}
          aria-label="Next page"
          title="Next page"
        >
          ▶
        </button>
        
        {/* Last */}
        <button
          className="pagination-button"
          onClick={handleLast}
          disabled={disabled || currentPage === totalPages}
          aria-label="Last page"
          title="Last page"
        >
          ⏭
        </button>
        
        {/* Jump to Page */}
        <div className="pagination-jump">
          <label htmlFor="page-jump" className="visually-hidden">
            Jump to page
          </label>
          <span className="pagination-jump-label">Go to:</span>
          <input
            id="page-jump"
            type="number"
            className="pagination-input"
            min={1}
            max={totalPages}
            value={currentPage}
            onChange={handlePageInput}
            disabled={disabled}
            aria-label="Jump to page"
          />
        </div>
      </div>
      
      {/* Page Size Selector */}
      {onPageSizeChange && (
        <div className="pagination-size">
          <label htmlFor="page-size-select" className="pagination-size-label">
            Per page:
          </label>
          <select
            id="page-size-select"
            className="pagination-select"
            value={itemsPerPage}
            onChange={handlePageSizeChange}
            disabled={disabled}
          >
            {pageSizeOptions.map(size => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}





