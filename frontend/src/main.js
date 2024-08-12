import "./assets/main.css";

import { createApp } from "vue";
import { createPinia } from "pinia";

import App from "./App.vue";
import router from "./router";
import "./assets/main.css";
import VueCameraLib from "vue-camera-lib";
import {
  SingleFileUpload,
  MultipleFileUpload,
} from "@canopassoftware/vue-file-upload";
import "@canopassoftware/vue-file-upload/style.css";

const app = createApp(App);

app.use(createPinia());
app.use(router);
app.use(VueCameraLib);

app.component("SingleFileUpload", SingleFileUpload);
app.component("MultipleFileUpload", MultipleFileUpload);

app.mount("#app");
