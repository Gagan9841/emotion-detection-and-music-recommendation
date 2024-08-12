<template>
  <div class="container mx-auto py-8">
    <h2 class="text-3xl font-bold text-center mb-6">Upload or Capture an Image</h2>
    <div class="flex justify-around items-center space-x-8">

      <!-- File Upload Section -->
      <div class="w-1/2 text-center">
        <h3 class="text-xl mb-4">Upload Image</h3>
        <div class="relative">
          <label
            class="flex justify-center items-center w-full h-48 px-4 transition bg-gray-100 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer hover:bg-gray-200 focus:outline-none">
            <span class="flex flex-col items-center space-y-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-12 h-12 text-gray-500" fill="none" viewBox="0 0 24 24"
                stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round"
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <span class="font-medium text-gray-500">
                Drop files to Attach, or
                <span class="text-blue-600 underline">browse</span>
              </span>
            </span>
            <input type="file" name="file_upload" class="hidden" @change="uploadFile($event)">
          </label>
        </div>
      </div>

      <!-- Camera Capture Section -->
      <div class="w-1/2 text-center">
        <h3 class="text-xl mb-4">Capture from Camera</h3>
        <WebCamUI :fullscreenState="false" @photoTaken="photoTaken" />
        <div v-if="image" class="mt-4">
          <img :src="image" alt="Captured" class="mx-auto rounded-lg shadow-lg" />
          <button @click="submitCapturedImage"
            class="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">Upload Captured Image</button>
        </div>
      </div>
    </div>

    <!-- Song Recommendations Section (Placeholder) -->
    <div class="mt-10">
      <h3 class="text-2xl text-center mb-4">Recommended Songs</h3>
      <!-- Add song recommendation code here -->
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { postRequest } from '../services/api';

const image = ref(null);
const songs = ref([]);

function displayRecommendations(emotion) {
  // Map the emotion to songs and update the songs array
  songs.value = getSongsForEmotion(emotion);
}

function photoTaken(data) {
  image.value = data.image_data_url;
}

async function uploadFile(event) {
  const file = event.target.files[0];
  if (file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await postRequest('/capture-emotion', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      alert(`Detected emotion: ${response.data.emotion}`);
    } catch (err) {
      console.error('Error uploading file:', err);
    }
  }
}

async function submitCapturedImage() {
  if (image.value) {
    const formData = new FormData();
    formData.append('image', image.value, 'captured_image.png');

    try {
      const response = await postRequest('/capture-emotion', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      alert(`Detected emotion: ${response.data.emotion}`);
    } catch (err) {
      console.error('Error submitting captured image:', err);
    }
  }
}

</script>

<style scoped>
.container {
  max-width: 900px;
}
</style>
