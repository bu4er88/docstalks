<template>
    <el-form @submit.prevent :inline="true" :model="FormInput" class="todo-form-input">
        <el-form-itim>
    </el-form>
</template>

<script lang="ts">
import { Options, Vue } from 'vue-class-component';

@Options({})
export default class TodoForm extends Vue {
    FormInput = { title: "", completed: false }

    onSubmit(){
        console.log("Todo", this.FormInput)
    }
}
</script> 