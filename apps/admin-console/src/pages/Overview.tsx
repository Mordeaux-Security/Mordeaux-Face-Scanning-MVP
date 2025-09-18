import { useQuery } from '@tanstack/react-query'
import { 
  Users, 
  Database, 
  AlertTriangle, 
  Activity,
  TrendingUp,
  Clock
} from 'lucide-react'

// Mock API functions
const fetchSystemStats = async () => {
  // Simulate API call
  await new Promise(resolve => setTimeout(resolve, 1000))
  return {
    totalTenants: 12,
    totalSources: 45,
    totalAlerts: 3,
    activeWorkers: 8,
    processedToday: 1250,
    avgProcessingTime: 2.3
  }
}

const fetchRecentActivity = async () => {
  await new Promise(resolve => setTimeout(resolve, 800))
  return [
    { id: 1, type: 'content_processed', message: 'New content processed from source "Website Crawler"', timestamp: '2 minutes ago' },
    { id: 2, type: 'face_detected', message: '5 faces detected in content ID: abc123', timestamp: '5 minutes ago' },
    { id: 3, type: 'alert_created', message: 'New alert: Suspicious activity detected', timestamp: '12 minutes ago' },
    { id: 4, type: 'policy_updated', message: 'Policy "Default Policy" updated by admin', timestamp: '1 hour ago' },
  ]
}

export default function Overview() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['system-stats'],
    queryFn: fetchSystemStats,
  })

  const { data: activity, isLoading: activityLoading } = useQuery({
    queryKey: ['recent-activity'],
    queryFn: fetchRecentActivity,
  })

  const statCards = [
    {
      name: 'Total Tenants',
      value: stats?.totalTenants || 0,
      icon: Users,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      name: 'Content Sources',
      value: stats?.totalSources || 0,
      icon: Database,
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    },
    {
      name: 'Active Alerts',
      value: stats?.totalAlerts || 0,
      icon: AlertTriangle,
      color: 'text-red-600',
      bgColor: 'bg-red-100',
    },
    {
      name: 'Active Workers',
      value: stats?.activeWorkers || 0,
      icon: Activity,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">System Overview</h1>
        <p className="mt-1 text-sm text-gray-500">
          Monitor your face protection system performance and activity
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {statCards.map((stat) => (
          <div key={stat.name} className="card">
            <div className="flex items-center">
              <div className={`flex-shrink-0 p-3 rounded-lg ${stat.bgColor}`}>
                <stat.icon className={`h-6 w-6 ${stat.color}`} />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">{stat.name}</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {statsLoading ? '...' : stat.value}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Performance Metrics</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <TrendingUp className="h-5 w-5 text-green-500 mr-2" />
                <span className="text-sm text-gray-600">Content Processed Today</span>
              </div>
              <span className="text-lg font-semibold text-gray-900">
                {statsLoading ? '...' : stats?.processedToday?.toLocaleString()}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Clock className="h-5 w-5 text-blue-500 mr-2" />
                <span className="text-sm text-gray-600">Avg Processing Time</span>
              </div>
              <span className="text-lg font-semibold text-gray-900">
                {statsLoading ? '...' : `${stats?.avgProcessingTime}s`}
              </span>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h3>
          <div className="space-y-3">
            {activityLoading ? (
              <div className="text-center py-4 text-gray-500">Loading activity...</div>
            ) : (
              activity?.map((item) => (
                <div key={item.id} className="flex items-start space-x-3">
                  <div className="flex-shrink-0">
                    <div className="h-2 w-2 bg-primary-500 rounded-full mt-2" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900">{item.message}</p>
                    <p className="text-xs text-gray-500">{item.timestamp}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <button className="btn btn-primary">
            Add New Source
          </button>
          <button className="btn btn-secondary">
            Create Policy
          </button>
          <button className="btn btn-secondary">
            View All Alerts
          </button>
        </div>
      </div>
    </div>
  )
}
