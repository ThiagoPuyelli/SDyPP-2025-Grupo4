import { Routes, Route } from "react-router-dom"
import Home from "./pages/Home";
import { Header } from "./components/Header";
import Transaccion from "./pages/Transaccion";

function App() {

  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/transaccion" element={<Transaccion />} />
      </Routes>
    </>
  );
}

export default App
