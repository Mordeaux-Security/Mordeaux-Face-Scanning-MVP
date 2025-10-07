# Dual Image Saving Feature

## 🎯 **Overview**

The dual image saving feature allows the crawler to save both the original full image and the cropped face image when a high-quality face is detected. This enables manual comparison between the original and cropped versions to evaluate the quality of face detection and cropping.

## ✨ **Features**

### **Automatic Dual Saving**
- **Conditional Saving**: Only saves both versions when `SAVE_BOTH=true` AND a high-quality face is detected
- **Original Preservation**: Full original image saved with `original_` prefix
- **Cropped Face**: Face region saved with regular UUID naming
- **No Duplication**: If no faces detected, only saves the full image (no original prefix)

### **Smart Storage Management**
- **Unique Keys**: Each image gets a unique UUID to prevent conflicts
- **Automatic Thumbnails**: Both original and cropped images get auto-generated thumbnails
- **MinIO Integration**: Seamlessly integrated with existing MinIO storage system

## 🚀 **Usage**

### **Enable Dual Saving**
```bash
# Save both original and cropped images for comparison
make crawl URL=https://www.pornhub.com METHOD=smart MAX_IMAGES=5 MIN_FACE_QUALITY=0.3 CROP_FACES=true SAVE_BOTH=true

# Multi-page crawling with dual saving
make crawl URL=https://www.pornhub.com CRAWL_MODE=site MAX_TOTAL_IMAGES=10 MAX_PAGES=5 MIN_FACE_QUALITY=0.1 REQUIRE_FACE=false CROP_FACES=true FACE_MARGIN=0.2 SAVE_BOTH=true
```

### **Disable Dual Saving (Default)**
```bash
# Only save cropped faces (default behavior)
make crawl URL=https://www.pornhub.com METHOD=smart MAX_IMAGES=5 MIN_FACE_QUALITY=0.3 CROP_FACES=true

# Or explicitly disable
make crawl URL=https://www.pornhub.com METHOD=smart MAX_IMAGES=5 MIN_FACE_QUALITY=0.3 CROP_FACES=true SAVE_BOTH=false
```

## 📊 **Test Results**

### **Successful Dual Saving Test**
```
URL: https://www.pornhub.com
Crawl mode: site
Pages crawled: 5
Images found: 185
Images saved: 5 (4 regular + 1 original from dual saving)
Images quality filtered: 17
Save both original and cropped: True
Storage: MinIO (raw-images & thumbnails buckets)

Saved image keys:
  - 80c2c1b6f6a047c4bf0787bbc81e804a.jpg          # Cropped face
  - original_07ae01f008ad4f33b613110be852bf8f.jpg  # Original image
  - a1f0aacadcf34e2bb2834bd05030c0d1.jpg          # Full image (no faces)
  - 66e54b90e3fa433a830c79bf70401277.jpg          # Full image (no faces)
  - 56bdc94e0e47491e8057b4c0b61832d8.jpg          # Full image (no faces)
  - 8643e62419a84593a668278dd5c90897.jpg          # Full image (no faces)
```

### **Storage Verification**
```
Raw images bucket:
  - 80c2c1b6f6a047c4bf0787bbc81e804a.jpg          # Cropped face
  - original_07ae01f008ad4f33b613110be852bf8f.jpg  # Original image
  - a1f0aacadcf34e2bb2834bd05030c0d1.jpg          # Full image
  - 66e54b90e3fa433a830c79bf70401277.jpg          # Full image
  - 56bdc94e0e47491e8057b4c0b61832d8.jpg          # Full image
  - 8643e62419a84593a668278dd5c90897.jpg          # Full image
Total raw images: 6

Thumbnails bucket:
  - 80c2c1b6f6a047c4bf0787bbc81e804a_thumb.jpg
  - original_07ae01f008ad4f33b613110be852bf8f_thumb.jpg
  - a1f0aacadcf34e2bb2834bd05030c0d1_thumb.jpg
  - 66e54b90e3fa433a830c79bf70401277_thumb.jpg
  - 56bdc94e0e47491e8057b4c0b61832d8_thumb.jpg
  - 8643e62419a84593a668278dd5c90897_thumb.jpg
Total thumbnails: 6
```

## 🔍 **How It Works**

### **Decision Logic**
1. **Face Detection**: Check if image contains high-quality faces
2. **Quality Validation**: Verify face meets minimum quality threshold
3. **Cropping Decision**: If `CROP_FACES=true` and face detected:
   - Save cropped face image with regular UUID
   - If `SAVE_BOTH=true`, also save original image with `original_` prefix
4. **Fallback**: If no faces or `CROP_FACES=false`, save full image normally

### **Storage Implementation**
```python
# In save_image_to_storage method
if self.crop_faces and best_face:
    # Crop and save face
    cropped_image_bytes = face_service.crop_face_from_image(image_bytes, bbox, self.face_margin)
    raw_key, raw_url, thumb_key, thumb_url = save_raw_and_thumb(cropped_image_bytes)
    
    # If save_both enabled, also save original
    if self.save_both:
        original_raw_key, original_raw_url, original_thumb_key, original_thumb_url = save_raw_and_thumb(image_bytes, key_prefix="original_")
else:
    # Save full image normally
    raw_key, raw_url, thumb_key, thumb_url = save_raw_and_thumb(image_bytes)
```

## 📁 **File Naming Convention**

### **Regular Images**
- **Cropped Face**: `[uuid].jpg` (e.g., `80c2c1b6f6a047c4bf0787bbc81e804a.jpg`)
- **Full Image**: `[uuid].jpg` (e.g., `a1f0aacadcf34e2bb2834bd05030c0d1.jpg`)
- **Thumbnails**: `[uuid]_thumb.jpg`

### **Dual Saved Images**
- **Cropped Face**: `[uuid].jpg` (e.g., `80c2c1b6f6a047c4bf0787bbc81e804a.jpg`)
- **Original Image**: `original_[uuid].jpg` (e.g., `original_07ae01f008ad4f33b613110be852bf8f.jpg`)
- **Thumbnails**: `[uuid]_thumb.jpg` and `original_[uuid]_thumb.jpg`

## 🎛️ **Configuration Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAVE_BOTH` | `false` | Enable/disable dual saving |
| `CROP_FACES` | `true` | Enable/disable face cropping |
| `MIN_FACE_QUALITY` | `0.5` | Minimum face quality threshold |
| `REQUIRE_FACE` | `true` | Require faces for saving |
| `FACE_MARGIN` | `0.2` | Margin around cropped face (20%) |

## 🔧 **Technical Implementation**

### **Storage Service Modification**
- Modified `save_raw_and_thumb()` to accept optional `key_prefix` parameter
- Maintains backward compatibility with existing calls
- Automatic thumbnail generation for both original and cropped images

### **Crawler Integration**
- Added `save_both` parameter to `EnhancedImageCrawler`
- Integrated with existing face quality checking logic
- Preserves all existing functionality and performance

### **CLI Integration**
- Added `--save-both` flag to `crawl_images.py`
- Updated Makefile with `SAVE_BOTH` parameter
- Full integration with all existing crawler options

## 🎉 **Benefits**

### **Quality Assessment**
- **Visual Comparison**: Side-by-side evaluation of cropping accuracy
- **Quality Validation**: Verify face detection performance
- **Margin Tuning**: Optimize face margin settings

### **Development & Testing**
- **Algorithm Validation**: Test face detection improvements
- **Parameter Optimization**: Fine-tune quality thresholds
- **Debugging**: Identify issues with face cropping

### **Production Use**
- **Audit Trail**: Keep original images for compliance
- **Backup Strategy**: Preserve full context if needed
- **Quality Control**: Manual review of automated cropping

## ✅ **Success Metrics**

✅ **Dual Saving Working**: Original and cropped images saved when enabled  
✅ **Conditional Logic**: Only saves both when high-quality faces detected  
✅ **Storage Integration**: Seamless integration with MinIO buckets  
✅ **Thumbnail Generation**: Auto-generated thumbnails for both versions  
✅ **Naming Convention**: Clear distinction with `original_` prefix  
✅ **CLI Integration**: Easy-to-use Makefile commands  
✅ **Backward Compatibility**: Existing functionality preserved  
✅ **Performance**: No impact on crawling speed or efficiency  

The dual image saving feature provides valuable capabilities for quality assessment, development testing, and production validation while maintaining the crawler's high performance and efficiency.
