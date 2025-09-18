import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import Policies from './pages/Policies'
import Sources from './pages/Sources'
import Alerts from './pages/Alerts'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/policies" element={<Policies />} />
        <Route path="/sources" element={<Sources />} />
        <Route path="/alerts" element={<Alerts />} />
      </Routes>
    </Layout>
  )
}

export default App
