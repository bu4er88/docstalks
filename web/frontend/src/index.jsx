import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { createRoot } from "react-dom";
import { ChakraProvider } from "@chakra-ui/react"

import Navbar from "./components/Header";
import Greeting from "./components/Greeting";
import Pricing from "./components/Pricing";
import Footer from "./components/Footer";
import ContactForm from "./components/ContactForm";


function home() {
  return (
    <ChakraProvider>
      <Navbar/>
      <Greeting/>
      <Pricing/>
      <Footer/>
    </ChakraProvider>
  );
};

function contact() {
  return (
    <ChakraProvider>
      <Navbar/>
      <ContactForm/>
      <Footer/>
    </ChakraProvider>
  );
};


function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={home()} />
        <Route path="/contact" element={contact()} />
      </Routes>
    </Router>
  );
}


const domNode = document.getElementById('root');
const root = createRoot(domNode);

root.render(<App/>);