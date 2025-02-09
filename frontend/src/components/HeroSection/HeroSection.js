import React, { useState, useRef, useEffect } from "react";
import "./HeroSection.css";
import { Link } from "react-router-dom";
import {
  Box,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  Stack,
  TextField,
  Typography,
  Divider,
} from "@mui/material";
import { IoMdCloseCircle } from "react-icons/io";
import axios from "axios";

const HeroSection = () => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const messageEndRef = useRef(null);

  const handleInputChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Add user message immediately
    setMessages((prevMessages) => [
      ...prevMessages,
      { text: query, sender: "user" },
    ]);
    setQuery(""); // Clear the input field after adding the user message

    // Show loading indicator
    setLoading(true);

    try {
      const res = await axios.post(
        "https://project-05-rate-professor-server.vercel.app/ask",
        {
          question: query,
        }
      );

      // Add bot response after loading
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: res.data.answer, sender: "bot" },
      ]);
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred. Please try again.");
    } finally {
      setLoading(false); // Stop the loading indicator
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  useEffect(() => {
    // Scroll to the bottom of the chat messages when new message is added
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <section className="hero">
      <div className="hero-content">
        <h1 className="hero-title">
          <span className="stroke-text">Empower</span>
          <span> Your </span>
          <span style={{ color: "var(--text-color)" }}>Education with AI-</span>
          <span>Powered Professor Ratings</span>
        </h1>
        <p className="hero-subtitle">
          Discover, rate, and share your experiences with professors using
          advanced AI insights.
        </p>
        <div className="cta-buttons">
          <button className="cta-button primary" onClick={handleClickOpen}>
            Rate a Professor Test
          </button>
          <Link to="/all-professors" className="cta-button secondary">
            Find the Best Professors
          </Link>
        </div>
      </div>
      <Dialog
        open={open}
        onClose={handleClose}
        fullWidth
        PaperProps={{
          sx: {
            overflowX: "hidden",
          },
        }}
      >
        <DialogTitle>
          Ask your question
          <IconButton
            edge="end"
            color="inherit"
            onClick={handleClose}
            sx={{
              position: "absolute",
              top: 8,
              right: 8,
            }}
          >
            <IoMdCloseCircle />
          </IconButton>
        </DialogTitle>
        <Box
          sx={{
            padding: 2,
            display: "flex",
            flexDirection: "column",
            height: "80vh",
            overflowX: "hidden",
          }}
        >
          <Divider sx={{ background: "white" }} />
          <Box
            sx={{ flex: 1, padding: 2, overflowY: "auto", overflowX: "hidden" }}
          >
            <Stack spacing={2}>
              {messages.map((msg, index) => (
                <Box
                  key={index}
                  sx={{
                    display: "flex",
                    justifyContent:
                      msg.sender === "user" ? "flex-end" : "flex-start",
                    flexDirection: "column",
                    alignItems:
                      msg.sender === "user" ? "flex-end" : "flex-start",
                    marginBottom: 2,
                  }}
                >
                  <Typography
                    variant="body1"
                    sx={{
                      wordWrap: "break-word",
                      backgroundColor:
                        msg.sender === "user" ? "#e0e0e0" : "#f5f5f5",
                      padding: 1,
                      borderRadius: 1,
                      maxWidth: "75%",
                      textAlign: msg.sender === "user" ? "right" : "left",
                    }}
                  >
                    {msg.text}
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      color: "gray",
                      marginTop: 0.5,
                      textAlign: msg.sender === "user" ? "right" : "left",
                    }}
                  >
                    {msg.sender === "user" ? "You" : "ProfPro"}
                  </Typography>
                </Box>
              ))}
              {loading && (
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "flex-start",
                    flexDirection: "column",
                    alignItems: "flex-start",
                    marginBottom: 2,
                  }}
                >
                  <Typography
                    variant="body1"
                    sx={{
                      wordWrap: "break-word",
                      backgroundColor: "#f5f5f5",
                      padding: 1,
                      borderRadius: 1,
                      maxWidth: "75%",
                      textAlign: "left",
                    }}
                  >
                    <span>
                      <div className="loader"></div>
                    </span>{" "}
                    {/* Loader */}
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      color: "gray",
                      marginTop: 0.5,
                      textAlign: "left",
                    }}
                  >
                    ProfPro
                  </Typography>
                </Box>
              )}
              <div ref={messageEndRef} /> {/* Scroll target */}
            </Stack>
          </Box>
          <Box
            sx={{
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
              marginTop: 2,
            }}
          >
            <TextField
              fullWidth
              label="Enter your query"
              value={query}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
            />
            <Button
              variant="contained"
              onClick={handleSubmit}
              sx={{ marginLeft: 1 }}
            >
              Send
            </Button>
          </Box>
        </Box>
      </Dialog>
    </section>
  );
};

export default HeroSection;

// import React, { useState, useRef, useEffect } from "react";
// import "./HeroSection.css";
// import { Link } from "react-router-dom";
// import {
//   Box,
//   Button,
//   IconButton,
//   Popover,
//   Stack,
//   TextField,
//   Typography,
//   Divider,
// } from "@mui/material";
// import { IoMdCloseCircle } from "react-icons/io";
// import axios from "axios";

// const HeroSection = () => {
//   const [anchorEl, setAnchorEl] = useState(null);
//   const [query, setQuery] = useState("");
//   const [response, setResponse] = useState("");
//   const [messages, setMessages] = useState([]);
//   const [loading, setLoading] = useState(false);

//   const messageEndRef = useRef(null);

//   const handleInputChange = (e) => {
//     setQuery(e.target.value);
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     // Add user message immediately
//     setMessages((prevMessages) => [
//       ...prevMessages,
//       { text: query, sender: "user" },
//     ]);
//     setQuery(""); // Clear the input field after adding the user message

//     // Show loading indicator
//     setLoading(true);

//     try {
//       const res = await axios.post("/ask", { question: query });

//       // Add bot response after loading
//       setMessages((prevMessages) => [
//         ...prevMessages,
//         { text: res.data.answer, sender: "bot" },
//       ]);
//     } catch (error) {
//       console.error("Error:", error);
//       alert("An error occurred. Please try again.");
//     } finally {
//       setLoading(false); // Stop the loading indicator
//     }
//   };

//   const handleKeyPress = (e) => {
//     if (e.key === "Enter") {
//       e.preventDefault();
//       handleSubmit();
//     }
//   };

//   const handleClick = (event) => {
//     setAnchorEl(event.currentTarget);
//   };

//   const handleClose = () => {
//     setAnchorEl(null);
//   };

//   const open = Boolean(anchorEl);
//   const id = open ? "simple-popover" : undefined;

//   useEffect(() => {
//     // Scroll to the bottom of the chat messages when new message is added
//     messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   }, [messages]);

//   return (
//     <section className="hero">
//       <div className="hero-content">
//         <h1 className="hero-title">
//           <span className="stroke-text">Empower</span>
//           <span> Your </span>
//           <span style={{ color: "var(--text-color)" }}>Education with AI-</span>
//           <span>Powered Professor Ratings</span>
//         </h1>
//         <p className="hero-subtitle">
//           Discover, rate, and share your experiences with professors using
//           advanced AI insights.
//         </p>
//         <div className="cta-buttons">
//           <button className="cta-button primary" onClick={handleClick}>
//             Rate a Professor Test
//           </button>
//           <Link to="/all-professors" className="cta-button secondary">
//             Find the Best Professors
//           </Link>
//         </div>
//       </div>
//       <Popover
//         id={id}
//         open={open}
//         anchorEl={anchorEl}
//         onClose={handleClose}
//         anchorOrigin={{
//           vertical: "bottom",
//           horizontal: "right",
//         }}
//         transformOrigin={{
//           vertical: "top",
//           horizontal: "right",
//         }}
//         PaperProps={{
//           sx: {
//             width: {
//               xs: "80vw",
//               sm: "40vw",
//               md: "30vw",
//             },
//             height: "80vh",
//             display: "flex",
//             flexDirection: "column",
//             background: "white",
//             color: "black",
//             overflow: "hidden", // Prevent scrollbar
//           },
//         }}
//       >
//         <IconButton
//           edge="end"
//           color="inherit"
//           onClick={handleClose}
//           sx={{
//             position: "absolute",
//             top: 8,
//             right: 8,
//           }}
//         >
//           <IoMdCloseCircle />
//         </IconButton>
//         <Box
//           sx={{
//             padding: 2,
//             display: "flex",
//             flexDirection: "column",
//             height: "100%",
//           }}
//         >
//           <Typography variant="h6" sx={{ textAlign: "center" }}>
//             Ask your question
//           </Typography>
//           <Divider sx={{ background: "white" }} />
//           <Box sx={{ flex: 1, padding: 2, overflowY: "auto" }}>
//             <Stack spacing={2}>
//               {messages.map((msg, index) => (
//                 <Box
//                   key={index}
//                   sx={{
//                     display: "flex",
//                     justifyContent:
//                       msg.sender === "user" ? "flex-end" : "flex-start",
//                     flexDirection: "column",
//                     alignItems:
//                       msg.sender === "user" ? "flex-end" : "flex-start",
//                     marginBottom: 2,
//                   }}
//                 >
//                   <Typography
//                     variant="body1"
//                     sx={{
//                       wordWrap: "break-word",
//                       backgroundColor:
//                         msg.sender === "user" ? "#e0e0e0" : "#f5f5f5",
//                       padding: 1,
//                       borderRadius: 1,
//                       maxWidth: "75%",
//                       textAlign: msg.sender === "user" ? "right" : "left",
//                     }}
//                   >
//                     {msg.text}
//                   </Typography>
//                   <Typography
//                     variant="caption"
//                     sx={{
//                       color: "gray",
//                       marginTop: 0.5,
//                       textAlign: msg.sender === "user" ? "right" : "left",
//                     }}
//                   >
//                     {msg.sender === "user" ? "You" : "ProfPro"}
//                   </Typography>
//                 </Box>
//               ))}
//               {loading && (
//                 <Box
//                   sx={{
//                     display: "flex",
//                     justifyContent: "flex-start",
//                     flexDirection: "column",
//                     alignItems: "flex-start",
//                     marginBottom: 2,
//                   }}
//                 >
//                   <Typography
//                     variant="body1"
//                     sx={{
//                       wordWrap: "break-word",
//                       backgroundColor: "#f5f5f5",
//                       padding: 1,
//                       borderRadius: 1,
//                       maxWidth: "75%",
//                       textAlign: "left",
//                     }}
//                   >
//                     <span>
//                       <div className="loader"></div>
//                     </span>{" "}
//                     {/* Loader */}
//                   </Typography>
//                   <Typography
//                     variant="caption"
//                     sx={{
//                       color: "gray",
//                       marginTop: 0.5,
//                       textAlign: "left",
//                     }}
//                   >
//                     ProfPro
//                   </Typography>
//                 </Box>
//               )}
//               <div ref={messageEndRef} /> {/* Scroll target */}
//             </Stack>
//           </Box>
//           <Box
//             sx={{
//               display: "flex",
//               flexDirection: "row",
//               alignItems: "center",
//               marginTop: 2,
//             }}
//           >
//             <TextField
//               fullWidth
//               label="Enter your query"
//               value={query}
//               onChange={handleInputChange}
//               onKeyPress={handleKeyPress}
//             />
//             <Button
//               variant="contained"
//               onClick={handleSubmit}
//               sx={{ marginLeft: 1 }}
//             >
//               Send
//             </Button>
//           </Box>
//         </Box>
//       </Popover>
//     </section>
//   );
// };

// export default HeroSection;
