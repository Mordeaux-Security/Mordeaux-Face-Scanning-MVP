const api = (import.meta.env.VITE_API_BASE || "") + "/api";
const app = document.getElementById("app");

console.log("🚀 Mordeaux Frontend Starting...");
console.log("📡 API Base URL:", api);

app.innerHTML = `
  <h1>Mordeaux — Face Search</h1>
  <form id="f">
    <div style="margin: 20px 0; padding: 20px; border: 2px dashed #ccc; border-radius: 8px; text-align: center; cursor: pointer; background: #f9f9f9;" onclick="document.getElementById('file').click()">
      <input type="file" id="file" accept="image/*" style="display: none;" />
      <div style="font-size: 18px; color: #666; margin-bottom: 10px;">
        📁 Click here to select a face image
      </div>
      <div style="font-size: 14px; color: #999;">
        Supports JPG, PNG files up to 10MB
      </div>
    </div>
    <div id="file-info" style="margin: 10px 0; padding: 10px; background: #e8f4fd; border-radius: 4px; display: none;">
      <strong>Selected:</strong> <span id="file-name"></span> (<span id="file-size"></span>)
    </div>
    <button type="submit" style="padding: 12px 24px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px;">
      🔍 Search for Similar Faces
    </button>
  </form>
  <pre id="out" style="background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; margin-top: 20px;"></pre>
  <img id="thumb" style="max-width:256px;display:none;margin-top: 20px;border-radius: 8px;box-shadow: 0 2px 8px rgba(0,0,0,0.1);"/>
`;

console.log("📝 Form elements created");

// Add file selection handler
document.getElementById("file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  const fileInfo = document.getElementById("file-info");
  const fileName = document.getElementById("file-name");
  const fileSize = document.getElementById("file-size");
  
  if (file) {
    console.log("📁 File selected:", file.name, file.size, "bytes");
    fileName.textContent = file.name;
    fileSize.textContent = (file.size / 1024 / 1024).toFixed(2) + " MB";
    fileInfo.style.display = "block";
  } else {
    fileInfo.style.display = "none";
  }
});

document.getElementById("f").addEventListener("submit", async (e) => {
  console.log("🔍 Form submitted!");
  e.preventDefault();
  
  const file = document.getElementById("file").files[0];
  console.log("📁 Selected file:", file);
  
  if (!file) {
    console.log("❌ No file selected");
    return;
  }
  
  console.log("📊 File details:", {
    name: file.name,
    size: file.size,
    type: file.type
  });
  
  const fd = new FormData();
  fd.append("file", file);
  console.log("📤 Sending request to:", api + "/search_face");
  
  try {
    const res = await fetch(api + "/search_face", { method: "POST", body: fd });
    console.log("📡 Response status:", res.status, res.statusText);
    
    if (!res.ok) {
      console.error("❌ HTTP Error:", res.status, res.statusText);
      const errorText = await res.text();
      console.error("❌ Error details:", errorText);
      document.getElementById("out").textContent = `Error ${res.status}: ${res.statusText}\n${errorText}`;
      return;
    }
    
    const json = await res.json();
    console.log("✅ Response received:", json);
    
    document.getElementById("out").textContent = JSON.stringify(json, null, 2);
    
    const img = document.getElementById("thumb");
    if (json.thumb_url) {
      console.log("🖼️ Setting thumbnail:", json.thumb_url);
      img.src = json.thumb_url;
      img.style.display = "block";
    } else {
      console.log("⚠️ No thumbnail URL in response");
    }
    
  } catch (error) {
    console.error("💥 Request failed:", error);
    document.getElementById("out").textContent = `Request failed: ${error.message}`;
  }
});

console.log("🎯 Event listeners attached");
