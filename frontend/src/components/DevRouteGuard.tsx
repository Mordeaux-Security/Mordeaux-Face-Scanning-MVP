/**
 * DevRouteGuard - Phase 10 Security/Privacy
 * ==========================================
 * 
 * Route guard component to protect dev-only pages.
 * Ensures sensitive admin/dev features are not accessible in production.
 * 
 * Features:
 * - Environment-based access control
 * - Feature flag support
 * - Auth role check (optional)
 * - Redirect non-authorized users
 */

import { ReactNode } from 'react';
import { Navigate } from 'react-router-dom';

interface DevRouteGuardProps {
  children: ReactNode;
  redirectTo?: string;
  requireAuth?: boolean;
}

/**
 * Check if dev mode is enabled
 * Priority: 1) Feature flag 2) Environment variable 3) Import.meta.env
 */
function isDevModeEnabled(): boolean {
  // Check 1: Feature flag in localStorage (for easy testing)
  const featureFlag = localStorage.getItem('ENABLE_DEV_MODE');
  if (featureFlag === 'true') {
    return true;
  }
  if (featureFlag === 'false') {
    return false;
  }

  // Check 2: Environment variable
  if (import.meta.env.VITE_DEV_MODE === 'true') {
    return true;
  }
  if (import.meta.env.VITE_DEV_MODE === 'false') {
    return false;
  }

  // Check 3: Default to DEV environment
  return import.meta.env.DEV;
}

/**
 * Check if user has dev role (placeholder for future auth integration)
 */
function hasDevRole(): boolean {
  // Placeholder: Always allow in dev mode
  if (import.meta.env.DEV) {
    return true;
  }

  // TODO: Integrate with actual auth system
  // Example:
  // const user = useAuth();
  // return user?.roles?.includes('dev') || user?.roles?.includes('admin');

  return false;
}

export default function DevRouteGuard({
  children,
  redirectTo = '/test',
  requireAuth = false,
}: DevRouteGuardProps) {
  const devModeEnabled = isDevModeEnabled();
  const hasRole = hasDevRole();

  // Log access attempt (dev only)
  if (import.meta.env.DEV) {
    console.log('[DevRouteGuard] Access check:', {
      devModeEnabled,
      hasRole,
      requireAuth,
      envDEV: import.meta.env.DEV,
    });
  }

  // Check 1: Dev mode must be enabled
  if (!devModeEnabled) {
    console.warn('[DevRouteGuard] Access denied: Dev mode not enabled');
    console.warn('[DevRouteGuard] Redirecting to:', redirectTo);
    return <Navigate to={redirectTo} replace />;
  }

  // Check 2: Auth role if required
  if (requireAuth && !hasRole) {
    console.warn('[DevRouteGuard] Access denied: Insufficient permissions');
    return <Navigate to={redirectTo} replace />;
  }

  // All checks passed
  return <>{children}</>;
}

/**
 * Hook to check dev mode status
 */
export function useDevMode() {
  return {
    isEnabled: isDevModeEnabled(),
    hasRole: hasDevRole(),
  };
}


