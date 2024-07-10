import { createApp } from 'vue'
import App from './App.vue'
import axios from 'axios'
import VueAxios from 'vue-axios'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import { Vue } from 'vue-class-component'


const app = createApp(App);

app.use(VueAxios, axios);
app.use(ElementPlus);
app.mount('#app');
