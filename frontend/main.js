const rawApiBase = import.meta.env.VITE_API_BASE || "/api";
const trimmedApiBase =
  rawApiBase.endsWith("/") && rawApiBase !== "/" ? rawApiBase.slice(0, -1) : rawApiBase;
const pipelineOverride = import.meta.env.VITE_PIPELINE_BASE;
const tenantId = window.TENANT_ID || "demo-tenant";
const app = document.getElementById("app");

const buildApiUrl = (path) => {
  if (trimmedApiBase.startsWith("http")) {
    return `${trimmedApiBase}${path}`;
  }
  return `${trimmedApiBase}${path}`;
};

const derivePipelineBase = () => {
  if (pipelineOverride) {
    return pipelineOverride.endsWith("/") && pipelineOverride !== "/"
      ? pipelineOverride.slice(0, -1)
      : pipelineOverride;
  }

  if (trimmedApiBase.includes("/api")) {
    return trimmedApiBase.replace(/\/api$/, "/pipeline");
  }

  if (trimmedApiBase.startsWith("http")) {
    return trimmedApiBase;
  }

  return "/pipeline";
};

const pipelineBase = derivePipelineBase();
const buildPipelineUrl = (path) => {
  if (pipelineBase.startsWith("http")) {
    return `${pipelineBase}${path}`;
  }
  return `${pipelineBase}${path}`;
};

console.log("🚀 Mordeaux Frontend Starting...");
console.log("📡 API Base URL:", trimmedApiBase);
console.log("🧠 Pipeline Base URL:", pipelineBase);

const state = {
  signedUp: false,
  identityId: null,
  email: null,
};

const layout = `
  <section>
    <h1>Mordeaux — Sign Up</h1>
    <p>Signing up automatically enrolls your identity so you can search the database afterward.</p>
    <form id="signup-form">
      <label style="display:block; margin:12px 0 6px;">Email</label>
      <input type="email" id="signup-email" required placeholder="you@example.com" style="width:100%;padding:10px;border:1px solid #ddd;border-radius:6px;">

      <label style="display:block; margin:12px 0 6px;">Password</label>
      <input type="password" id="signup-password" required minlength="8" placeholder="Minimum 8 characters" style="width:100%;padding:10px;border:1px solid #ddd;border-radius:6px;">

      <label style="display:block; margin:12px 0 6px;">Identity ID</label>
      <input type="text" id="identityId" required placeholder="e.g., person-123" style="width:100%;padding:10px;border:1px solid #ddd;border-radius:6px;">

      <div style="margin: 20px 0; padding: 20px; border: 2px dashed #ccc; border-radius: 8px; text-align: center; cursor: pointer; background: #f9f9f9;" onclick="document.getElementById('signup-images').click()">
        <input type="file" id="signup-images" accept="image/*" multiple style="display: none;" />
        <div style="font-size: 18px; color: #666; margin-bottom: 10px;">
          📁 Click here to select 3-5 face images
        </div>
        <div style="font-size: 14px; color: #999;">
          Minimum 3 images required, recommended 3-5. Supports JPG, PNG files up to 10MB each.
        </div>
      </div>

      <div id="signup-file-info" style="margin: 10px 0; padding: 10px; background: #e8f4fd; border-radius: 4px; display: none;">
        <strong>Selected:</strong> <span id="file-count"></span> images (<span id="file-size"></span>)
      </div>

      <button type="submit" style="padding: 12px 24px; font-size: 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px;">
        ✅ Create Account & Enroll Identity
      </button>
    </form>
    <pre id="signup-output" style="background:#f5f5f5;padding:15px;border-radius:4px;overflow-x:auto;margin-top:20px;"></pre>
  </section>

  <section id="search-section" style="margin-top:40px; display:none;">
    <h2>🔍 Search Your Face</h2>
    <p>Upload a selfie to search across the database. Available after successful signup.</p>
    <form id="search-form">
      <label style="display:block; margin:12px 0 6px;">Search image</label>
      <input type="file" id="search-image" accept="image/*" style="display:block;margin-bottom:12px;" />
      <button type="submit" style="padding: 10px 20px; background:#28a745;color:#fff;border:none;border-radius:4px;cursor:pointer;">🔍 Search Now</button>
    </form>
    <div id="search-status" style="margin-top:15px; padding:12px; border-radius:6px; background:#f0f0f0;"></div>
    <div id="search-results" style="margin-top:20px; display:grid; grid-template-columns:repeat(auto-fill, minmax(200px, 1fr)); gap:16px;"></div>
  </section>
`;

app.innerHTML = layout;

const fileInput = document.getElementById("signup-images");
const fileInfo = document.getElementById("signup-file-info");
const fileCount = document.getElementById("file-count");
const fileSize = document.getElementById("file-size");
const signupOutput = document.getElementById("signup-output");
const searchSection = document.getElementById("search-section");
const searchStatus = document.getElementById("search-status");
const searchResults = document.getElementById("search-results");

// Helper to render search results as image cards
const renderSearchResults = (hits) => {
  searchResults.innerHTML = "";
  
  if (!hits || hits.length === 0) {
    searchResults.innerHTML = '<p style="grid-column:1/-1; text-align:center; color:#666;">No matches found</p>';
    return;
  }
  
  hits.forEach((hit, idx) => {
    const similarity = hit.similarity_pct || Math.round(hit.score * 100);
    const imageUrl = hit.image_url || hit.thumb_url || hit.crop_url;
    const site = hit.payload?.site || "unknown";

    let badgeColor = "#dc3545";
    if (similarity >= 90) badgeColor = "#28a745";
    else if (similarity >= 75) badgeColor = "#ffc107";

    const card = document.createElement("div");
    card.style.cssText = "background:white; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.1); overflow:hidden; transition:transform 0.2s,box-shadow 0.2s;";
    card.onmouseenter = () => { card.style.transform = "translateY(-4px)"; card.style.boxShadow = "0 6px 20px rgba(0,0,0,0.15)"; };
    card.onmouseleave = () => { card.style.transform = "translateY(0)"; card.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)"; };

    const imageHtml = imageUrl
      ? '<img src="' + imageUrl + '" alt="Match ' + (idx + 1) + '" style="width:100%;height:100%;object-fit:cover;" onerror="this.style.display=\'none\'">'
      : '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#999;">No image</div>';

    card.innerHTML =
      '<div style="position:relative;aspect-ratio:1;background:#f0f0f0;">' +
        imageHtml +
        '<div style="position:absolute;top:8px;right:8px;background:' + badgeColor + ';color:white;padding:4px 10px;border-radius:20px;font-weight:bold;font-size:14px;">' +
          similarity + '%' +
        '</div>' +
        '<div style="position:absolute;bottom:8px;left:8px;background:rgba(0,0,0,0.7);color:white;padding:3px 8px;border-radius:4px;font-size:11px;">' +
          '#' + (idx + 1) +
        '</div>' +
      '</div>' +
      '<div style="padding:12px;">' +
        '<div style="font-weight:600;color:#333;font-size:14px;margin-bottom:4px;">Match ' + (idx + 1) + '</div>' +
        '<div style="color:#666;font-size:12px;">Site: ' + site + '</div>' +
        '<div style="margin-top:8px;background:#f5f5f5;padding:6px 8px;border-radius:4px;font-size:11px;color:#888;">' +
          'Score: ' + (hit.score ? hit.score.toFixed(4) : "N/A") +
        '</div>' +
      '</div>';

    searchResults.appendChild(card);
  });
};

const toBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.includes(",") ? reader.result.split(",")[1] : reader.result;
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

const summarizeFiles = (files) => {
  const imageFiles = files.filter((file) => file.type.startsWith("image/"));
  if (imageFiles.length === 0) {
    fileInfo.style.display = "none";
    return [];
  }

  const limited = imageFiles.slice(0, 5);
  const sizeMb = limited.reduce((sum, file) => sum + file.size, 0) / 1024 / 1024;

  fileCount.textContent =
    imageFiles.length > 5 ? `${limited.length} of ${imageFiles.length}` : `${limited.length}`;
  fileSize.textContent = `${sizeMb.toFixed(2)} MB`;
  fileInfo.style.display = "block";

  if (imageFiles.length < 3) {
    fileCount.textContent += " (need at least 3)";
  }

  return limited;
};

fileInput.addEventListener("change", (event) => {
  summarizeFiles(Array.from(event.target.files));
});

document.getElementById("signup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  signupOutput.textContent = "";

  const email = document.getElementById("signup-email").value.trim();
  const password = document.getElementById("signup-password").value;
  const identityId = document.getElementById("identityId").value.trim();
  const selectedFiles = summarizeFiles(Array.from(fileInput.files));

  if (!email || !password || !identityId) {
    signupOutput.textContent = "Please fill in email, password, and identity ID.";
    return;
  }

  if (selectedFiles.length < 3) {
    signupOutput.textContent = "Please select at least 3 valid face images.";
    return;
  }

  signupOutput.textContent = "Uploading photos and creating your account...";

  try {
    const imagesB64 = await Promise.all(selectedFiles.map((file) => toBase64(file)));
    const payload = {
      tenant_id: tenantId,
      email,
      password,
      identity_id: identityId,
      images_b64: imagesB64,
    };

    const response = await fetch(buildApiUrl("/v1/signup"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-ID": tenantId,
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      signupOutput.textContent = `Signup failed (${response.status}): ${errorText}`;
      console.error("Signup error:", errorText);
      return;
    }

    const json = await response.json();
    console.log("✅ Signup response:", json);
    signupOutput.textContent = JSON.stringify(json, null, 2);

    state.signedUp = true;
    state.identityId = identityId;
    state.email = email;

    searchSection.style.display = "block";
    searchStatus.innerHTML = `<span style="color:#28a745;">✅ Signup complete!</span> Upload a selfie below to search for matches in the database.`;
    searchResults.innerHTML = "";
  } catch (err) {
    console.error("Signup failed:", err);
    signupOutput.textContent = `Signup failed: ${err.message}`;
  }
});

document.getElementById("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!state.signedUp) {
    searchStatus.innerHTML = `<span style="color:#dc3545;">⚠️ Please complete signup before searching.</span>`;
    return;
  }

  const file = document.getElementById("search-image").files[0];
  if (!file || !file.type.startsWith("image/")) {
    searchStatus.innerHTML = `<span style="color:#dc3545;">⚠️ Please choose a face photo to search.</span>`;
    return;
  }

  searchStatus.innerHTML = `<span style="color:#007bff;">🔄 Searching for matches...</span>`;
  searchResults.innerHTML = "";

  try {
    const imageB64 = await toBase64(file);
    const payload = {
      tenant_id: tenantId,
      image_b64: imageB64,
      top_k: 10,
      threshold: 0.10,  // Lower threshold - typical similar faces score 0.16-0.22
    };

    const response = await fetch(buildApiUrl("/v1/search"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let errorDetail = "";
      try {
        const errorJson = await response.json();
        errorDetail = errorJson.detail || errorJson.error || JSON.stringify(errorJson);
        // If it's a quality rejection, show helpful message
        if (errorJson.detail?.error === "no_usable_faces") {
          const reasons = errorJson.detail.reasons || [];
          const numFaces = errorJson.detail.num_faces_detected || 0;
          searchStatus.innerHTML = `<span style="color:#dc3545;">❌ Face quality check failed. Detected ${numFaces} face(s) but none met quality requirements. Reasons: ${reasons.join(', ')}</span>`;
        } else {
          searchStatus.innerHTML = `<span style="color:#dc3545;">❌ Search failed (${response.status}): ${errorDetail}</span>`;
        }
      } catch (e) {
        const errorText = await response.text();
        searchStatus.innerHTML = `<span style="color:#dc3545;">❌ Search failed (${response.status}): ${errorText}</span>`;
      }
      console.error("Search error:", errorDetail);
      return;
    }

    const json = await response.json();
    console.log("🔎 Search results:", json);
    console.log("🔎 Response structure:", {
      hasCount: 'count' in json,
      count: json.count,
      hasHits: 'hits' in json,
      hitsLength: json.hits?.length,
      keys: Object.keys(json)
    });

    // Defensive: ensure count exists (fallback to hits length if missing)
    const count = json.count ?? json.hits?.length ?? 0;
    const hits = json.hits ?? [];

    if (count === 0) {
      searchStatus.innerHTML = `<span style="color:#856404; background:#fff3cd; padding:8px 12px; border-radius:4px;">
        🔍 No matches found above threshold. Try a clearer photo or different angle.
      </span>`;
      searchResults.innerHTML = "";
      return;
    }

    searchStatus.innerHTML = `<span style="color:#28a745;">✅ Found ${count} match${count > 1 ? 'es' : ''}!</span>
      <span style="color:#666; font-size:13px;"> (showing faces with ≥${Math.round(payload.threshold * 100)}% similarity)</span>`;
    
    renderSearchResults(hits);
  } catch (err) {
    console.error("Search failed:", err);
    searchStatus.innerHTML = `<span style="color:#dc3545;">❌ Search failed: ${err.message}</span>`;
  }
});

console.log("🎯 Event listeners attached");
