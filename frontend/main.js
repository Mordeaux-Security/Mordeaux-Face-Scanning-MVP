const api = (import.meta.env.VITE_API_BASE || "http://localhost:8001") + "/api";
const app = document.getElementById("app");

console.log("🚀 Mordeaux Frontend Starting...");
console.log("📡 API Base URL:", api);

app.innerHTML = `
  <h1>Mordeaux — Identity Enrollment</h1>
  <form id="f">
    <div style="margin: 20px 0; padding: 20px; border: 2px dashed #ccc; border-radius: 8px; text-align: center; cursor: pointer; background: #f9f9f9;" onclick="document.getElementById('file').click()">
      <input type="file" id="file" accept="image/*" multiple style="display: none;" />
      <div style="font-size: 18px; color: #666; margin-bottom: 10px;">
        📁 Click here to select 3-5 face images
      </div>
      <div style="font-size: 14px; color: #999;">
        Minimum 3 images required, recommended 3-5. Supports JPG, PNG files up to 10MB each
      </div>
    </div>
    <div id="file-info" style="margin: 10px 0; padding: 10px; background: #e8f4fd; border-radius: 4px; display: none;">
      <strong>Selected:</strong> <span id="file-count"></span> images (<span id="file-size"></span>)
    </div>
    <div id="identity-input" style="display: none; margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;">
      <label for="identityId" style="display: block; margin-bottom: 5px; font-weight: 600; color: #555;">Identity ID:</label>
      <input type="text" id="identityId" placeholder="Enter identity identifier (e.g., person-123)" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px;">
      <div style="font-size: 12px; color: #666; margin-top: 5px;">
        💡 Unique identifier for this person's identity
      </div>
    </div>
    <button type="submit" style="padding: 12px 24px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px;">
      ✅ Enroll Identity
    </button>
  </form>
  <pre id="out" style="background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; margin-top: 20px;"></pre>
`;

console.log("📝 Form elements created");

// Add file selection handler
document.getElementById("file").addEventListener("change", (e) => {
  const files = Array.from(e.target.files);
  const fileInfo = document.getElementById("file-info");
  const fileCount = document.getElementById("file-count");
  const fileSize = document.getElementById("file-size");
  const identityInput = document.getElementById("identity-input");
  
  if (files.length > 0) {
    // Filter to only image files
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
      console.log("❌ No valid image files selected");
      fileInfo.style.display = "none";
      identityInput.style.display = "none";
      return;
    }
    
    // Validate count
    if (imageFiles.length < 3) {
      console.log(`⚠️ Need at least 3 images, got ${imageFiles.length}`);
      fileCount.textContent = `${imageFiles.length} (need at least 3)`;
    } else if (imageFiles.length > 5) {
      console.log(`⚠️ Too many images, using first 5 of ${imageFiles.length}`);
      fileCount.textContent = `5 of ${imageFiles.length} (max 5 recommended)`;
    } else {
      fileCount.textContent = `${imageFiles.length}`;
    }
    
    const totalSize = imageFiles.reduce((sum, file) => sum + file.size, 0);
    fileSize.textContent = (totalSize / 1024 / 1024).toFixed(2) + " MB";
    fileInfo.style.display = "block";
    
    if (imageFiles.length >= 3) {
      identityInput.style.display = "block";
    } else {
      identityInput.style.display = "none";
    }
  } else {
    fileInfo.style.display = "none";
    identityInput.style.display = "none";
  }
});

document.getElementById("f").addEventListener("submit", async (e) => {
  console.log("✅ Enrollment form submitted!");
  e.preventDefault();
  
  const files = Array.from(document.getElementById("file").files);
  const imageFiles = files.filter(file => file.type.startsWith('image/'));
  
  if (imageFiles.length < 3) {
    console.log("❌ Need at least 3 images");
    document.getElementById("out").textContent = `Error: Please select at least 3 images (currently: ${imageFiles.length})`;
    return;
  }
  
  const identityId = document.getElementById("identityId").value.trim();
  if (!identityId) {
    console.log("❌ Identity ID required");
    document.getElementById("out").textContent = "Error: Please enter an Identity ID";
    return;
  }
  
  // Limit to 5 images if more selected
  const filesToUse = imageFiles.slice(0, 5);
  
  console.log("📁 Selected files:", filesToUse.length, "images");
  console.log("👤 Identity ID:", identityId);
  
  // Get tenant ID from environment or use default
  const tenantId = window.TENANT_ID || "demo-tenant";
  
  // Convert images to base64
  const imagesB64 = await Promise.all(
    filesToUse.map(file => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          // Remove data:image/...;base64, prefix
          const base64 = reader.result.split(',')[1];
          resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    })
  );
  
  // Build request body
  const requestBody = {
    tenant_id: tenantId,
    identity_id: identityId,
    images_b64: imagesB64,
    overwrite: true
  };
  
  // Build URL - use pipeline API for enrollment
  const pipelineApi = (import.meta.env.VITE_API_BASE || "http://localhost:8001");
  const url = new URL(pipelineApi + "/api/v1/enroll_identity");
  
  console.log("📤 Sending enrollment request to:", url.toString());
  
  try {
    const res = await fetch(url.toString(), { 
      method: "POST",
      headers: {
        'Content-Type': 'application/json',
        'X-Tenant-ID': tenantId
      },
      body: JSON.stringify(requestBody)
    });
    console.log("📡 Response status:", res.status, res.statusText);
    
    if (!res.ok) {
      console.error("❌ HTTP Error:", res.status, res.statusText);
      const errorText = await res.text();
      console.error("❌ Error details:", errorText);
      document.getElementById("out").textContent = `Error ${res.status}: ${res.statusText}\n${errorText}`;
      return;
    }
    
    const json = await res.json();
    console.log("✅ Enrollment response received:", json);
    
    document.getElementById("out").textContent = JSON.stringify(json, null, 2);
    
  } catch (error) {
    console.error("💥 Request failed:", error);
    document.getElementById("out").textContent = `Request failed: ${error.message}`;
  }
});

console.log("🎯 Event listeners attached");
