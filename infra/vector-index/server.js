const express = require('express');
const app = express();
const port = 8080;

// In-memory storage for vectors
const vectors = new Map();
let nextId = 1;

app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'vector-index' });
});

// Upsert vector
app.post('/upsert', (req, res) => {
  const { id, vector, metadata } = req.body;
  
  if (!vector || !Array.isArray(vector) || vector.length !== 512) {
    return res.status(400).json({ error: 'Invalid vector: must be 512-dimensional array' });
  }
  
  const vectorId = id || `vector_${nextId++}`;
  vectors.set(vectorId, { vector, metadata: metadata || {} });
  
  res.json({ id: vectorId, status: 'upserted' });
});

// Query vectors
app.post('/query', (req, res) => {
  const { vector, top_k = 10, filter } = req.body;
  
  if (!vector || !Array.isArray(vector) || vector.length !== 512) {
    return res.status(400).json({ error: 'Invalid vector: must be 512-dimensional array' });
  }
  
  // Simple cosine similarity calculation
  const results = [];
  for (const [id, data] of vectors.entries()) {
    const similarity = cosineSimilarity(vector, data.vector);
    results.push({
      id,
      score: similarity,
      metadata: data.metadata
    });
  }
  
  // Sort by similarity and return top_k
  results.sort((a, b) => b.score - a.score);
  res.json({ results: results.slice(0, top_k) });
});

// Helper function for cosine similarity
function cosineSimilarity(a, b) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

app.listen(port, '0.0.0.0', () => {
  console.log(`Vector index service running on port ${port}`);
});
