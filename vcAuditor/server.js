const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();
const port = 3000;
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules')));
app.use(express.json());
app.use(express.static('public'));

let audioFolderPath = '';
let textFolderPath = '';

app.post('/set-folders', (req, res) => {
  audioFolderPath = req.body.audioFolder;
  textFolderPath = req.body.textFolder;
  res.send('Folders set successfully');
});

// Define the /list-audio and /list-text routes before the custom middleware
app.get('/list-audio', (req, res) => {
  if (!audioFolderPath) return res.status(404).send('Audio folder not set');
  fs.readdir(audioFolderPath, (err, files) => {
    if (err) return res.status(500).send('Error reading files');
    res.json(files.filter(file => file.endsWith('.wav')));
  });
});

app.get('/list-text', (req, res) => {
    if (!textFolderPath) {
      res.status(400).send('Text folder path not set');
      return;
    }
  
    fs.readdir(textFolderPath, (err, files) => {
      if (err) {
        console.log('Error reading text folder:', err); // Log the error for debugging
        res.status(500).send('Error reading text files');
        return;
      }
      res.json(files.filter(file => file.endsWith('.txt')));
    });
  });

app.use('/audio', (req, res, next) => {
  if (audioFolderPath) {
    express.static(audioFolderPath)(req, res, next);
  } else {
    res.status(404).send('Audio folder not set');
  }
});

app.use('/text', (req, res, next) => {
  if (textFolderPath) {
    express.static(textFolderPath)(req, res, next);
  } else {
    res.status(404).send('Text folder not set');
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});