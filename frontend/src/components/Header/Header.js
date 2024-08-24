import React, { useState } from "react";
import "./Header.css";
import logo from "../../Image/logo.png";
import { Link } from "react-router-dom";

const Header = () => {
  const [isNavOpen, setIsNavOpen] = useState(false);

  const toggleNav = () => {
    setIsNavOpen(!isNavOpen);
  };

  return (
    <div>
      <header className="header">
        <div className="logo">
          <img
            src={logo}
            alt="AI"
            className="logo-img"
            style={{
              width: "100px",
            }}
          />
          <h1>AI Rate My Professor</h1>
        </div>
        <button className="nav-toggle" onClick={toggleNav}>
          ☰
        </button>
        <nav className={`nav ${isNavOpen ? "active" : ""}`}>
          <ul className="nav-list">
            <li className="nav-item">
              <Link
                to="/"
                style={{
                  textDecoration: "none",
                }}
              >
                Home
              </Link>
            </li>

            <li className="nav-item">
              <Link
                to="/all-professors"
                style={{
                  textDecoration: "none",
                }}
              >
                Professors
              </Link>
            </li>
            <li className="nav-item">
              <a href="#signup" className="cta-button secondary">
                Sign Up
              </a>
            </li>
            <li className="nav-item">
              <a href="#login">Login</a>
            </li>
          </ul>
        </nav>
      </header>
    </div>
  );
};

export default Header;
