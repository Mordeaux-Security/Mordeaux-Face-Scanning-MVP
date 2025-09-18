import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, Eye, CheckCircle, XCircle, Clock, Filter } from 'lucide-react'

// Mock API functions
const fetchAlerts = async () => {
  await new Promise(resolve => setTimeout(resolve, 1000))
  return [
    {
      id: '1',
      type: 'suspicious_activity',
      severity: 'high',
      title: 'Multiple face matches detected',
      description: 'Face matching algorithm detected potential security threat with 95% confidence',
      tenant: 'Enterprise Corp',
      status: 'open',
      createdAt: '2024-01-15T10:30:00Z',
      data: {
        confidence: 0.95,
        faceCount: 3,
        source: 'Website Crawler'
      }
    },
    {
      id: '2',
      type: 'policy_violation',
      severity: 'medium',
      title: 'Access attempt blocked',
      description: 'User attempted to access restricted content based on current policy',
      tenant: 'Default Tenant',
      status: 'acknowledged',
      createdAt: '2024-01-15T09:15:00Z',
      data: {
        userId: 'user123',
        contentId: 'content456',
        policy: 'Strict Monitoring'
      }
    },
    {
      id: '3',
      type: 'system_error',
      severity: 'low',
      title: 'Face detection service timeout',
      description: 'Face detection service experienced timeout during processing',
      tenant: 'Startup Inc',
      status: 'resolved',
      createdAt: '2024-01-15T08:45:00Z',
      data: {
        service: 'face-detector',
        timeout: 30,
        retryCount: 3
      }
    },
    {
      id: '4',
      type: 'data_quality',
      severity: 'medium',
      title: 'Low quality face detected',
      description: 'Face detection found image with quality below threshold',
      tenant: 'Enterprise Corp',
      status: 'open',
      createdAt: '2024-01-15T07:20:00Z',
      data: {
        quality: 0.3,
        threshold: 0.5,
        contentId: 'content789'
      }
    }
  ]
}

export default function Alerts() {
  const [filter, setFilter] = useState('all')
  const [selectedAlert, setSelectedAlert] = useState<any>(null)

  const { data: alerts, isLoading } = useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
  })

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100'
      case 'high': return 'text-orange-600 bg-orange-100'
      case 'medium': return 'text-yellow-600 bg-yellow-100'
      case 'low': return 'text-blue-600 bg-blue-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return 'bg-red-100 text-red-800'
      case 'acknowledged': return 'bg-yellow-100 text-yellow-800'
      case 'resolved': return 'bg-green-100 text-green-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'open': return XCircle
      case 'acknowledged': return Clock
      case 'resolved': return CheckCircle
      default: return AlertTriangle
    }
  }

  const filteredAlerts = alerts?.filter(alert => {
    if (filter === 'all') return true
    return alert.status === filter
  })

  const alertStats = {
    total: alerts?.length || 0,
    open: alerts?.filter(a => a.status === 'open').length || 0,
    acknowledged: alerts?.filter(a => a.status === 'acknowledged').length || 0,
    resolved: alerts?.filter(a => a.status === 'resolved').length || 0,
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Alerts</h1>
        <p className="mt-1 text-sm text-gray-500">
          Monitor system alerts and security notifications
        </p>
      </div>

      {/* Alert Stats */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-4">
        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-3 rounded-lg bg-gray-100">
              <AlertTriangle className="h-6 w-6 text-gray-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Total Alerts</p>
              <p className="text-2xl font-semibold text-gray-900">{alertStats.total}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-3 rounded-lg bg-red-100">
              <XCircle className="h-6 w-6 text-red-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Open</p>
              <p className="text-2xl font-semibold text-gray-900">{alertStats.open}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-3 rounded-lg bg-yellow-100">
              <Clock className="h-6 w-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Acknowledged</p>
              <p className="text-2xl font-semibold text-gray-900">{alertStats.acknowledged}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-3 rounded-lg bg-green-100">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-500">Resolved</p>
              <p className="text-2xl font-semibold text-gray-900">{alertStats.resolved}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Filter className="h-5 w-5 text-gray-400" />
            <div className="flex space-x-2">
              {['all', 'open', 'acknowledged', 'resolved'].map((status) => (
                <button
                  key={status}
                  onClick={() => setFilter(status)}
                  className={`px-3 py-1 rounded-full text-sm font-medium capitalize ${
                    filter === status
                      ? 'bg-primary-100 text-primary-800'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {status}
                </button>
              ))}
            </div>
          </div>
          <button className="btn btn-primary">
            Mark All as Read
          </button>
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-4">
        {isLoading ? (
          <div className="text-center py-8 text-gray-500">Loading alerts...</div>
        ) : (
          filteredAlerts?.map((alert) => {
            const StatusIcon = getStatusIcon(alert.status)
            return (
              <div key={alert.id} className="card">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0">
                      <div className={`p-2 rounded-lg ${getSeverityColor(alert.severity).split(' ')[1]}`}>
                        <AlertTriangle className={`h-5 w-5 ${getSeverityColor(alert.severity).split(' ')[0]}`} />
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <h3 className="text-lg font-medium text-gray-900">{alert.title}</h3>
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(alert.severity)}`}>
                          {alert.severity}
                        </span>
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(alert.status)}`}>
                          <StatusIcon className="h-3 w-3 mr-1" />
                          {alert.status}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-gray-600">{alert.description}</p>
                      <div className="mt-2 flex items-center space-x-4 text-xs text-gray-500">
                        <span>Tenant: {alert.tenant}</span>
                        <span>Type: {alert.type.replace('_', ' ')}</span>
                        <span>Created: {new Date(alert.createdAt).toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setSelectedAlert(alert)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                    {alert.status === 'open' && (
                      <button className="btn btn-secondary text-sm">
                        Acknowledge
                      </button>
                    )}
                    {alert.status === 'acknowledged' && (
                      <button className="btn btn-primary text-sm">
                        Resolve
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* Alert Details Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75" onClick={() => setSelectedAlert(null)} />
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="flex items-center mb-4">
                  <AlertTriangle className="h-6 w-6 text-primary-600 mr-2" />
                  <h3 className="text-lg font-medium text-gray-900">
                    {selectedAlert.title}
                  </h3>
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Description</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedAlert.description}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Severity</label>
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(selectedAlert.severity)}`}>
                        {selectedAlert.severity}
                      </span>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Status</label>
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedAlert.status)}`}>
                        {selectedAlert.status}
                      </span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Alert Data</label>
                    <div className="mt-2 p-3 bg-gray-50 rounded-lg">
                      <pre className="text-xs text-gray-600 overflow-auto">
                        {JSON.stringify(selectedAlert.data, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <button
                  onClick={() => setSelectedAlert(null)}
                  className="btn btn-primary"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
