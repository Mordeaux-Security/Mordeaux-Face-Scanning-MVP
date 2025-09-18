import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Plus, Edit, Trash2, Shield, AlertCircle } from 'lucide-react'

// Mock API functions
const fetchPolicies = async () => {
  await new Promise(resolve => setTimeout(resolve, 1000))
  return [
    {
      id: '1',
      name: 'Default Policy',
      tenant: 'Default Tenant',
      rules: [
        { type: 'allow', conditions: {}, actions: ['search', 'view'] }
      ],
      isActive: true,
      createdAt: '2024-01-15T10:00:00Z',
      updatedAt: '2024-01-15T10:00:00Z'
    },
    {
      id: '2',
      name: 'Strict Monitoring',
      tenant: 'Enterprise Corp',
      rules: [
        { type: 'block', conditions: { confidence: { $lt: 0.8 } }, actions: ['block'] },
        { type: 'alert', conditions: { confidence: { $gte: 0.8, $lt: 0.95 } }, actions: ['alert'] }
      ],
      isActive: true,
      createdAt: '2024-01-10T14:30:00Z',
      updatedAt: '2024-01-12T09:15:00Z'
    },
    {
      id: '3',
      name: 'Permissive Policy',
      tenant: 'Startup Inc',
      rules: [
        { type: 'allow', conditions: {}, actions: ['search', 'view', 'download'] }
      ],
      isActive: false,
      createdAt: '2024-01-05T16:45:00Z',
      updatedAt: '2024-01-08T11:20:00Z'
    }
  ]
}

export default function Policies() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedPolicy, setSelectedPolicy] = useState<any>(null)

  const { data: policies, isLoading } = useQuery({
    queryKey: ['policies'],
    queryFn: fetchPolicies,
  })

  const getRuleTypeColor = (type: string) => {
    switch (type) {
      case 'allow': return 'text-green-600 bg-green-100'
      case 'block': return 'text-red-600 bg-red-100'
      case 'alert': return 'text-yellow-600 bg-yellow-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getRuleTypeIcon = (type: string) => {
    switch (type) {
      case 'allow': return '✓'
      case 'block': return '✗'
      case 'alert': return '⚠'
      default: return '?'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Policies</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage access control policies for your tenants
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn btn-primary flex items-center"
        >
          <Plus className="h-4 w-4 mr-2" />
          Create Policy
        </button>
      </div>

      {/* Policies Table */}
      <div className="card">
        {isLoading ? (
          <div className="text-center py-8 text-gray-500">Loading policies...</div>
        ) : (
          <div className="overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Policy
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Tenant
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Rules
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Updated
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {policies?.map((policy) => (
                  <tr key={policy.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Shield className="h-5 w-5 text-gray-400 mr-3" />
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {policy.name}
                          </div>
                          <div className="text-sm text-gray-500">
                            ID: {policy.id}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {policy.tenant}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex flex-wrap gap-1">
                        {policy.rules.slice(0, 2).map((rule: any, index: number) => (
                          <span
                            key={index}
                            className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getRuleTypeColor(rule.type)}`}
                          >
                            {getRuleTypeIcon(rule.type)} {rule.type}
                          </span>
                        ))}
                        {policy.rules.length > 2 && (
                          <span className="text-xs text-gray-500">
                            +{policy.rules.length - 2} more
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        policy.isActive 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {policy.isActive ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(policy.updatedAt).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex items-center justify-end space-x-2">
                        <button
                          onClick={() => setSelectedPolicy(policy)}
                          className="text-primary-600 hover:text-primary-900"
                        >
                          <Edit className="h-4 w-4" />
                        </button>
                        <button className="text-red-600 hover:text-red-900">
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Policy Details Modal */}
      {selectedPolicy && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75" onClick={() => setSelectedPolicy(null)} />
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="flex items-center mb-4">
                  <Shield className="h-6 w-6 text-primary-600 mr-2" />
                  <h3 className="text-lg font-medium text-gray-900">
                    {selectedPolicy.name}
                  </h3>
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Tenant</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedPolicy.tenant}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Rules</label>
                    <div className="mt-2 space-y-2">
                      {selectedPolicy.rules.map((rule: any, index: number) => (
                        <div key={index} className="p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center justify-between">
                            <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getRuleTypeColor(rule.type)}`}>
                              {getRuleTypeIcon(rule.type)} {rule.type}
                            </span>
                          </div>
                          <div className="mt-2 text-xs text-gray-600">
                            <div><strong>Conditions:</strong> {JSON.stringify(rule.conditions)}</div>
                            <div><strong>Actions:</strong> {rule.actions.join(', ')}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <button
                  onClick={() => setSelectedPolicy(null)}
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
