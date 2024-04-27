import React from "react";
import { createRoot } from "react-dom";
import { ChakraProvider } from "@chakra-ui/react"

import Navbar from "./components/Header";
import Greeting from "./components/Greeting";
import Pricing from "./components/Pricing";
import Footer from "./components/Footer";



function App() {
  return (
    <ChakraProvider>
      <Navbar/>
      <Greeting/>
      <Pricing/>
      <Footer/>
    </ChakraProvider>
  );
};


const domNode = document.getElementById('root');
const root = createRoot(domNode);

root.render(<App/>);