import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Plus, Edit, Trash2, Database, Globe, Upload, AlertCircle } from 'lucide-react'

// Mock API functions
const fetchSources = async () => {
  await new Promise(resolve => setTimeout(resolve, 1000))
  return [
    {
      id: '1',
      name: 'Website Crawler',
      type: 'crawler',
      url: 'https://example.com',
      tenant: 'Default Tenant',
      status: 'active',
      lastRun: '2024-01-15T10:30:00Z',
      contentCount: 1250,
      config: { interval: '1h', depth: 3 }
    },
    {
      id: '2',
      name: 'API Upload',
      type: 'api',
      url: 'https://api.example.com/upload',
      tenant: 'Enterprise Corp',
      status: 'active',
      lastRun: '2024-01-15T10:25:00Z',
      contentCount: 890,
      config: { auth: 'bearer', rate_limit: 100 }
    },
    {
      id: '3',
      name: 'Manual Upload',
      type: 'upload',
      url: null,
      tenant: 'Startup Inc',
      status: 'inactive',
      lastRun: '2024-01-14T15:45:00Z',
      contentCount: 45,
      config: { max_size: '10MB', allowed_types: ['jpg', 'png'] }
    }
  ]
}

export default function Sources() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedSource, setSelectedSource] = useState<any>(null)

  const { data: sources, isLoading } = useQuery({
    queryKey: ['sources'],
    queryFn: fetchSources,
  })

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'crawler': return Globe
      case 'api': return Database
      case 'upload': return Upload
      default: return Database
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'crawler': return 'text-blue-600 bg-blue-100'
      case 'api': return 'text-green-600 bg-green-100'
      case 'upload': return 'text-purple-600 bg-purple-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800'
      case 'inactive': return 'bg-gray-100 text-gray-800'
      case 'error': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Content Sources</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage your content sources and crawling configurations
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn btn-primary flex items-center"
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Source
        </button>
      </div>

      {/* Sources Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
        {isLoading ? (
          <div className="col-span-full text-center py-8 text-gray-500">Loading sources...</div>
        ) : (
          sources?.map((source) => {
            const TypeIcon = getTypeIcon(source.type)
            return (
              <div key={source.id} className="card">
                <div className="flex items-start justify-between">
                  <div className="flex items-center">
                    <div className={`flex-shrink-0 p-2 rounded-lg ${getTypeColor(source.type).split(' ')[1]}`}>
                      <TypeIcon className={`h-5 w-5 ${getTypeColor(source.type).split(' ')[0]}`} />
                    </div>
                    <div className="ml-3">
                      <h3 className="text-lg font-medium text-gray-900">{source.name}</h3>
                      <p className="text-sm text-gray-500">{source.tenant}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setSelectedSource(source)}
                      className="text-gray-400 hover:text-gray-600"
                    >
                      <Edit className="h-4 w-4" />
                    </button>
                    <button className="text-gray-400 hover:text-red-600">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                <div className="mt-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Type</span>
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getTypeColor(source.type)}`}>
                      {source.type}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Status</span>
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(source.status)}`}>
                      {source.status}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Content Count</span>
                    <span className="text-sm font-medium text-gray-900">
                      {source.contentCount.toLocaleString()}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Last Run</span>
                    <span className="text-sm text-gray-900">
                      {new Date(source.lastRun).toLocaleDateString()}
                    </span>
                  </div>

                  {source.url && (
                    <div className="pt-2 border-t border-gray-200">
                      <span className="text-sm text-gray-600">URL</span>
                      <p className="text-sm text-gray-900 truncate">{source.url}</p>
                    </div>
                  )}
                </div>

                <div className="mt-4 flex space-x-2">
                  <button className="flex-1 btn btn-secondary text-sm">
                    View Content
                  </button>
                  <button className="flex-1 btn btn-primary text-sm">
                    {source.status === 'active' ? 'Pause' : 'Start'}
                  </button>
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* Source Details Modal */}
      {selectedSource && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75" onClick={() => setSelectedSource(null)} />
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="flex items-center mb-4">
                  {(() => {
                    const TypeIcon = getTypeIcon(selectedSource.type)
                    return <TypeIcon className="h-6 w-6 text-primary-600 mr-2" />
                  })()}
                  <h3 className="text-lg font-medium text-gray-900">
                    {selectedSource.name}
                  </h3>
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Configuration</label>
                    <div className="mt-2 p-3 bg-gray-50 rounded-lg">
                      <pre className="text-xs text-gray-600 overflow-auto">
                        {JSON.stringify(selectedSource.config, null, 2)}
                      </pre>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Status</label>
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedSource.status)}`}>
                        {selectedSource.status}
                      </span>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700">Content Count</label>
                      <p className="text-sm text-gray-900">{selectedSource.contentCount.toLocaleString()}</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <button
                  onClick={() => setSelectedSource(null)}
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
