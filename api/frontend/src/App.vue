<template>
  <el-container>
    <el-main>
      <todo-list></todo-list>
    </el-main>
  </el-container>
</template>

<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import TodoList from './components/TodoList.vue';

@Options({
  components: {
    TodoList,
  },
})
export default class App extends Vue {}
</script>

<style> 
body {
  margin: 0;
}

.el-container {
  background-color: #e9eef3;
  color: var(--el-text-color-primary);
  text-align: center;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
