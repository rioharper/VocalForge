import WaveSurfer from "https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js"
import RegionsPlugin from "./node_modules/wavesurfer.js/dist/plugins/regions.esm.js"
import Hover from "https://unpkg.com/wavesurfer.js@7/dist/plugins/hover.esm.js"
import Minimap from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/minimap.esm.js'
let activeTab = null
let showAll = true
class CommonFunctions {
  constructor() {
    // Initialize common properties if needed
  }

  // Common functions can be defined here
  // ...
}

class WaveformGenerator extends CommonFunctions {
  waveformDiv
  file
  wavesurfer
  regions

  constructor(waveformDiv, file) {
    super()
    this.loadingAnimation()
    this.waveformDiv = waveformDiv
    this.file = file
    this.wavesurfer = this.initializeWaveSurfer()
    this.regions = this.registerRegionsPlugin()
    this.setupInteractions()
  }

  loadingAnimation(){
    const waveformsDiv = document.getElementById("waveforms");

    const options = {
      root: null,
      rootMargin: '0px',
      threshold: 0.5
    };

    const observer = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const numLoadingBars = 120;
          for (let i = 0; i < numLoadingBars; i++) {
            const height = Math.floor(Math.random() * 200) + 60;
            const loadingBar = document.createElement("div");
            loadingBar.className = "loading-bar";
            loadingBar.style.left = `${(i * 2) + 2}%`; // Adjust position as needed
            loadingBar.style.animationDelay = `${i * 0.1}s`; // Add delay for staggered effect
            loadingBar.style.height = `${height}px`;
            waveformsDiv.appendChild(loadingBar);
          }
          observer.unobserve(entry.target);
          setTimeout(() => {
            const loadingBars = document.querySelectorAll('.loading-bar');
            loadingBars.forEach(bar => bar.remove());
          }, numLoadingBars * 20);
        }
      });
    }, options);
  observer.observe(waveformsDiv);
  }

  initializeWaveSurfer() {
    return WaveSurfer.create({
      container: this.waveformDiv,
      url: `/audio/${this.file}`,
      autoscroll: true,
      autoCenter: true,
      minPxPerSec: 100,
      height: 600,
      barGap: 5,
      barWidth: 6,
      barRadius: 30,
      progessColor: "#BF0603",
      plugins: [
        Hover.create({
          lineColor: "#ff0000",
          lineWidth: 6,
          labelBackground: "#555",
          labelColor: "#fff",
          labelSize: "15px",
        }),
        Minimap.create({
          height: 100,
          waveColor: '#ddd',
          progressColor: '#708D81',
          // the Minimap takes all the same options as the WaveSurfer itself
        }),
      ],
    })
  }

  registerRegionsPlugin() {
    return this.wavesurfer.registerPlugin(RegionsPlugin.create())
  }

  setupInteractions() {  
    
  }
}

class RegionsManager {
  median;
  stdDev;
  activeRegion;
  constructor(regions, text) {
    this.regions = regions;
    this.text = text;
    this.lines = [];
    this.createRegions();
    this.setupInteractions();
  }

  calculateStats(lines) {
    //add error handling
    const values = lines.map(line => parseFloat(line.split(" ")[2])).filter(value => !isNaN(value));
    const mean = values.reduce((acc, val) => acc + val) / values.length;
    const stdDev = Math.sqrt(values.reduce((acc, val) => acc + (val - mean) ** 2, 0) / values.length);
    const median = values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
    return { stdDev, median };
  }

  createRegions() {
    this.text.then(text => {
      const lines = text.split("\n");
      const stats = this.calculateStats(lines);
      this.median = stats.median;
      this.stdDev = stats.stdDev;
      let index = 0;
      lines.forEach(line => {
        this.addRegionFromLine(line, index);
        index++;
      });
    });
  }

  setupInteractions() {
    this.regions.on("region-clicked", (region, e) => {
      if (e.shiftKey) {
        e.stopPropagation();
        this.activeRegion = region;
        region.play();
      } else {
        this.activeRegion = null;
      }
    });

    this.regions.on("region-in", function(region) {
      this.activeRegion = region;
      const newid = region.id.filter(id => id !== "inactive").concat("active")
      region.setOptions({ id: newid})
      const targetrow = activeTab.tableGenerator.table.querySelector(`tr[data-index="${region.id[0]}"]`);
      targetrow.style.display = 'table-row';
      targetrow.scrollIntoView({ block: 'start' , behavior: 'smooth'});
      if (!showAll) {
        const allTableRows = activeTab.tableGenerator.table.querySelectorAll('tbody tr');
        allTableRows.forEach(row => {
          if (row.dataset.index !== region.id[0]) {
            row.style.display = 'none';
          }
        });
      }else{
        const tableDiv = document.querySelector(`.table-div${activeTab.index}`)
        const row = activeTab.tableGenerator.table.querySelector(`tr[data-index="${region.id[0]}"]`);
        if (row) {
          //set scroll behavior to smoothly scroll to the row
          tableDiv.scrollTo({
            top: row.offsetTop - tableDiv.offsetTop,
            behavior: 'smooth'
          });
        }
      }
    }.bind(this))

    this.regions.on("region-out", function(region){
      const newid = region.id.filter(id => id !== "active").concat("inactive")
      region.setOptions({ id: newid})
    }.bind(this))
  }

  addRegionFromLine(line, index) {
    const [start, end, confidence, content] = this.extractLineData(line);
    
    if (isNaN(start) || isNaN(end)) {
      return;
    }
    const region = this.regions.addRegion({
      id: [index.toString(), "inactive"],
      start: parseFloat(start),
      end: parseFloat(end),
      color: this.getConfidenceColor(confidence),
    });
  
    const checkButton = document.createElement("button");
    checkButton.innerHTML = "&#10003;"; // Checkmark icon
    checkButton.addEventListener("click", () => {
      //remove inactive and active from id
      const newid = region.id.filter(id => id !== "inactive" && id !== "active").concat("checked")
      region.setOptions({id: newid})
      console.log(region.id)
      const row = document.querySelector(`tr[data-index="${region.id[0]}"]`);
      row.querySelector("td:nth-child(5)").textContent = "verified";

    });
    // set to the left most side of the element
    checkButton.style.right = "10px";
    const garbageButton = document.createElement("button");
    garbageButton.innerHTML = "&#128465;";
    garbageButton.addEventListener("click", () => {
      region.setOptions({ id: region.id.concat("removed")})
      const row = document.querySelector(`tr[data-index="${region.id[0]}"]`);
      row.remove();
    });
    checkButton.style.right = "20px";
  
    region.element.appendChild(checkButton);
    region.element.appendChild(garbageButton);
  }
  extractLineData(line) {
    const [floats, , text2] = line.split(" | ");
    const [start, end, confidence] = floats.split(" ");
    return [start, end, confidence, text2];
  }

  getConfidenceColor(value) {
    const logistic = x => 1 / (1 + Math.exp(-x * 2));
    const red = Math.max(0, Math.min(255, Math.round(logistic((this.median - value) / this.stdDev) * 255)));
    const green = Math.max(0, Math.min(255, Math.round(logistic((value - this.median) / this.stdDev) * 255 * 0.8)));
    const blue = 0;
    const alpha = 0.4;
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }
}


class TableGenerator {
  waveformDiv
  processedLines
  sortSelect
  table
  text
  constructor(tableDiv, text, waveformGenerator) {
    this.text = text
    this.tableDiv = tableDiv
    this.waveformGenerator = waveformGenerator
    this.WaveSurfer = waveformGenerator.wavesurfer
    this.regions = waveformGenerator.regions
    this.processedLines = []
    this.sortSelect = null
    this.table = this.createTable()

  }

  createExportButton() {
    const exportButton = document.createElement("button");
    exportButton.textContent = "Export";
    exportButton.addEventListener("click", () => this.exportTable());
    return exportButton;
  }

  createShowAllButton() {
    const ShowAllButton = document.createElement("button");
    ShowAllButton.textContent = "Show Active";
    ShowAllButton.addEventListener("click", () => {
      if (ShowAllButton.textContent === "Show Active") {
        console.log("should only show active")
        showAll = false;
        ShowAllButton.textContent = "Show All";
      }
      else{
        console.log("should show all")
        showAll = true;
        ShowAllButton.textContent = "Show Active";
        //show all rows
        const allTableRows = activeTab.tableGenerator.table.querySelectorAll('tr');
        allTableRows.forEach(row => {
          if (row.querySelector("th") == null) {
            row.style.display = 'table-row';
          }
        })
      }
    });
    return ShowAllButton;
  }

 

  exportTable() {
    let data = '';
    //get rows as an array but skip the last row (the one with the delete button)
    const rows = Array.from(this.table.querySelectorAll("tbody tr"));
    rows.forEach((row, index) => {
      const cells = Array.from(row.querySelectorAll("td"));
      const text1 = cells[0].querySelector("textarea").value;
      const start = cells[1].textContent.replace("Start: ", "");
      const end = cells[2].textContent.replace("End: ", "");
      const confidence = cells[4].textContent === "verified" ? -0.0 : cells[3].textContent.replace("Confidence: ", "");
      data += `${start} ${end} ${confidence} | ${text1}`;
    });
    const blob = new Blob([data], {type: 'text/plain'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = this.waveformGenerator.file.replace(".wav", "") + "_segmented.txt";
    link.href = url;
    link.click();
  }
  
  processTextLines(textLines) {
    return textLines.map(this.parseTextLine)
  }

  parseTextLine(line) {
    const [floats, text1, text2, text3] = line.split(" | ")
    const [start, end, confidence] = floats.split(" ")
    if (isNaN(start) || isNaN(end)) {
      return null
    }
    return {
      texts: [text1, text2, text3],
      start: parseFloat(start).toFixed(1),
      end: parseFloat(end).toFixed(1),
      confidence: parseFloat(confidence).toFixed(2),
    }
  }

  createSortSelect() {
    const sortBasisSelect = document.createElement("select")
    const sortBasisOptions = ["Sort by Start Time", "Sort by Confidence"]
    sortBasisOptions.forEach(option => {
      const sortOption = document.createElement("option")
      sortOption.value = option
      sortOption.text = option
      sortBasisSelect.appendChild(sortOption)
    })
  
    const sortDirectionSelect = document.createElement("select")
    const sortDirectionOptions = ["Ascending", "Descending"]
    sortDirectionOptions.forEach(option => {
      const sortOption = document.createElement("option")
      sortOption.value = option
      sortOption.text = option
      sortDirectionSelect.appendChild(sortOption)
    })
    
    return [sortBasisSelect, sortDirectionSelect]
  }

  createTableHeader(columns) {
    const thead = document.createElement("thead")
    const tr = document.createElement("tr")
    // Add other column headers
    columns.forEach(column => {
      const th = document.createElement("th")
      th.textContent = column
      tr.appendChild(th)
    })
    thead.appendChild(tr)
    return thead
  }

  createTableBody() {
    const tbody = document.createElement("tbody");
    this.processedLines.forEach((line, index) => {
      if (line === null) {
        return;
      }
      const tr = document.createElement("tr");
      tr.setAttribute("data-index", index); // add index as a data attribute
  
      // Create the first column with the text area
      const td1 = document.createElement("td");
      const textarea = document.createElement("textarea");
      textarea.textContent = line.texts[1];
      // set font size and font family
      textarea.style.fontSize = "20px";
      textarea.style.fontFamily = "Helvetica";
      textarea.rows = 7; // add this line to set the number of rows
      textarea.cols = 30;
      td1.appendChild(textarea);
      tr.appendChild(td1);
  
      // Create the second column with the start time
      const td2 = document.createElement("td");
      td2.textContent = "Start: " + line.start;
      tr.appendChild(td2);
  
      // Create the third column with the end time
      const td3 = document.createElement("td");
      td3.textContent = "End: " + line.end;
      tr.appendChild(td3);
  
      // Create the fourth column with the confidence
      const td4 = document.createElement("td");
      td4.textContent = "Confidence: " + line.confidence;
      tr.appendChild(td4);
  
      // Add delete button to row
      const deleteButton = document.createElement("button");
      deleteButton.textContent = "Delete";
      deleteButton.addEventListener("click", () => {
        const region = this.regions.getRegions()[index];
        region.setOptions({ id: [index.toString(), "removed"] });
        tr.remove();
      });
  
      const td5 = document.createElement("td");
      td5.appendChild(deleteButton);
      tr.appendChild(td5);
  
      // Add event listener for row click
      tr.addEventListener("click", function(event){
        const startTime = tr.querySelector("td:nth-child(3)").innerHTML.replace("Start: ", "");
        this.waveformGenerator.wavesurfer.setTime(startTime);
        this.waveformGenerator.wavesurfer.play();
      }.bind(this));
  
      tbody.appendChild(tr);
    });
  
    this.regions.on("region-updated", function(region) {
      const start = region.start;
      const end = region.end;
      const regionId = parseInt(region.id[0]);
      const row = this.table.querySelector(`tr[data-index="${regionId}"]`);
      row.querySelector("td:nth-child(2)").textContent = "Start: " + start.toFixed(1);
      row.querySelector("td:nth-child(3)").textContent = "End: " + end.toFixed(1);
    }.bind(this));
  
    return tbody;
  }

  createTableElement() {
    const tbody = this.createTableBody()
    const table = document.createElement("table")
    table.appendChild(tbody)
    return table
  }

  sortTableRows(table, sortOption, sortDirection) {
    const tbody = table.querySelector("tbody")
    // Obtain rows as a JavaScript Array
    const rows = Array.from(tbody.querySelectorAll("tr"))

    rows.sort((a, b) => {
      const aVal = parseFloat(a.querySelector(`td:nth-child(${sortOption === "Sort by Confidence" ? 4 : (sortOption === "Sort by Start" ? 2 : 3)})`).textContent.split(":")[1].trim());
      const bVal = parseFloat(b.querySelector(`td:nth-child(${sortOption === "Sort by Confidence" ? 4 : (sortOption === "Sort by Start" ? 2 : 3)})`).textContent.split(":")[1].trim());
      return sortDirection === "Ascending" ? aVal - bVal : bVal - aVal;
    });
  
    // Empty tbody
    while(tbody.firstChild) {
      tbody.firstChild.remove()
    }
    // Add sorted rows back to tbody
    rows.forEach(row => {
      tbody.appendChild(row)
    })

  }

  createTable() {
    this.text.then(text => {
      const textLines = text.split("\n");
      this.processedLines = this.processTextLines(textLines).filter(line => line !== null);
      [this.sortBasisSelect, this.sortDirectionSelect] = this.createSortSelect();
      this.table = this.createTableElement();
      const sortingButtonsDiv = document.createElement("div"); // create new div for buttons and options
      sortingButtonsDiv.className = "sortingButtons"; // add class to new div
      sortingButtonsDiv.appendChild(this.sortBasisSelect); // append sortBasisSelect to new div
      sortingButtonsDiv.appendChild(this.sortDirectionSelect); // append sortDirectionSelect to new div

      const buttonsDiv = document.createElement("div"); // create new div for buttons and options
      buttonsDiv.className = "buttons"; // add class to new div
      buttonsDiv.appendChild(this.createExportButton());
      buttonsDiv.appendChild(this.createShowAllButton());
      this.tableDiv.appendChild(sortingButtonsDiv); // append new div to tableDiv
      this.tableDiv.appendChild(buttonsDiv); // append new div to tableDiv
      this.tableDiv.appendChild(this.table); // append table to tableDiv
      // Call sortTableRows function on change event of sortBasisSelect or sortDirectionSelect
      this.sortBasisSelect.addEventListener("change", () => {
        this.sortTableRows(this.table, this.sortBasisSelect.value, this.sortDirectionSelect.value);
      });
      this.sortDirectionSelect.addEventListener("change", () => {
        this.sortTableRows(this.table, this.sortBasisSelect.value, this.sortDirectionSelect.value);
      });
      console.debug('Table created successfully and appended to tableDiv');
    });
  }
}

class Tab {
  audioFile
  textFile
  index
  tab
  waveformDiv
  regions
  tableGenerator
  waveformGenerator
  isSelected = false

  constructor(audioFile, textFile, index, tabs) {
    this.audioFile = audioFile
    this.index = index
    this.tabs = tabs
    this.textFile = textFile
    this.text = this.getTextFromFile(textFile)
    this.tab = document.createElement("div")
    this.tab.className = "tab"
    this.tab.textContent = audioFile
    this.tab.onclick = () => this.selectTab(this.index)
    
    this.waveformDiv = document.createElement("div")
    this.waveformDiv.className = "waveform"
    this.tableDiv = document.createElement("div")
    this.tableDiv.className = `table-div${index}` 
    this.waveformGenerator = new WaveformGenerator(this.waveformDiv, audioFile)
    this.regions = new RegionsManager(this.waveformGenerator.regions, this.text, this.waveformGenerator.wavesurfer)
    this.tableGenerator = new TableGenerator(this.tableDiv, this.text, this.waveformGenerator)
    this.setupInteractions()
    // initialize the tablediv animation
    this.tableDiv.style.display = "none"
  }

  selectTab(index) {
    this.tabs.forEach((tab, i) => {
      if (i === index) {
        tab.select()
        tab.isSelected = true // Add this line
        activeTab = tab
        const tabDiv = document.querySelector(`#tabs > div:nth-child(${index + 1})`)
        console.log(tabDiv)
        tabDiv.style.backgroundColor = "#001427"
        tabDiv.style.border = "2px solid white"
        tabDiv.style.color = "#ffffff"
      } else {
        tab.deselect()
        tab.isSelected = false // Add this line
        const tabDiv = document.querySelector(`#tabs > div:nth-child(${i + 1})`)
        console.log(tabDiv)
        tabDiv.style.backgroundColor = "#ffffff"
        tabDiv.style.border = "none"
        tabDiv.style.color = "#000000"
      }
    })
  }

  setupInteractions() {
    document.addEventListener("keydown", (event) => {
      if (event.code === "Space" && this.isSelected) {
        if (this.waveformGenerator.wavesurfer.isPlaying()) {
          this.waveformGenerator.wavesurfer.pause()
        } else {
          this.waveformGenerator.wavesurfer.play()
        }
      }
    })
    const tablesDiv = document.querySelector('#tables');
    let isDragging = false;
    let initialX;
    let initialY;
    let currentX = 0;
    let currentY = 0;

    tablesDiv.addEventListener('mousedown', (event) => {
      isDragging = true;
      initialX = event.clientX;
      initialY = event.clientY;
    });

    function updatePosition() {
      tablesDiv.style.transform = `translate(${currentX}px, ${currentY}px)`;
      if (isDragging) {
        requestAnimationFrame(updatePosition);
      }
    }

    tablesDiv.addEventListener('mousemove', (event) => {
      if (isDragging) {
        currentX = event.clientX - initialX;
        currentY = event.clientY - initialY;
        requestAnimationFrame(updatePosition);
      }
    });

    tablesDiv.addEventListener('mouseup', () => {
      isDragging = false;
    });
  }

  getTextFromFile(file) {
    return fetch(`/text/${file}`).then(res => res.text())
  }

  select() {
    this.waveformDiv.style.display = "block"
    this.tableDiv.style.display = "block"
  }

  deselect() {
    this.waveformDiv.style.display = "none"
    this.waveformGenerator.wavesurfer.pause()
    this.tableDiv.style.display = "none"

  }
}

class TabManager extends CommonFunctions {
  tabsDiv
  waveformsDiv
  audioFiles
  textFiles
  tabs
  constructor() {
    super()
    this.tabsDiv = document.getElementById("tabs")
    this.waveformsDiv = document.getElementById("waveforms")
    this.tableDiv = document.getElementById("tables")
    this.audioFiles = []
    this.textFiles = []
    this.tabs = []
    this.setupInteractions()

  }

  createTab(audioFile, index) {
    const tab = new Tab(audioFile, this.textFiles[index], index, this.tabs)
    this.tabs.push(tab)
    this.tabsDiv.appendChild(tab.tab)
    console.log('Tab ' + audioFile + ' created successfully')
    this.waveformsDiv.appendChild(tab.waveformDiv)
    this.tableDiv.appendChild(tab.tableDiv)
  }

  selectTab(index) {
    this.tabs.forEach((tab, i) => {
      if (i === index) {
        tab.select()
        activeTab = tab
        const tabDiv = document.querySelector(`#tabs > div:nth-child(${index + 1})`)
        console.log(tabDiv)
        tabDiv.style.backgroundColor = "#001427"
        tabDiv.style.border = "2px solid white"
        tabDiv.style.color = "#ffffff"
      } else {
        tab.deselect()       
      }
    })
  }

  setupInteractions() {
    document.addEventListener("keydown", (event) => {
      //if user presses space and is not in the tables div
      if (event.code === "Space" && !event.target.closest("#tables")) {
        event.preventDefault();
      }
    });

    const tabs = document.querySelectorAll('.tab');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        // Remove active class from all tabs
        tabs.forEach(tab => {
          tab.classList.remove('active');
        });

        // Add active class to clicked tab
        tab.classList.add('active');
      });
    });

    // const slider = document.createElement("input")
    // slider.type = "range"
    // slider.min = "10"
    // slider.max = "200"
    // slider.value = "100"
    // slider.oninput = (e) => {
    //   const minPxPerSec = Number(e.target.value)
    //   this.activeTab.waveformGenerator.wavesurfer.zoom(minPxPerSec)
    //     }
    // //set slider to be centered
    // slider.style.marginLeft = "600px"
    // slider.style.display = "visible"
    // const zoomText = document.createElement("p")
    // zoomText.textContent = "Zoom"
    // zoomText.style.marginLeft = "550px"
    // zoomText.style.display = "inline-block"
    // slider.appendChild(zoomText)
    // this.tabsDiv.appendChild(slider)  
  }


  async setFolders() {
    const audioFolder = document.getElementById('audioFolder').value
    const textFolder = document.getElementById('textFolder').value

    try {
      const responseMessage = await this.sendFolderSettings(audioFolder, textFolder)
      alert(responseMessage)

      const { audioFiles, textFiles } = await this.fetchAudioAndTextFiles()
      if (!audioFiles || !textFiles) {
        console.error("Received empty response from fetching data")
        return
      }

      this.audioFiles = audioFiles
      this.textFiles = textFiles

      this.tabsDiv.innerHTML = ""
      this.waveformsDiv.innerHTML = ""
      this.tabs = []

      audioFiles.forEach((audioFile, index) => this.createTab(audioFile, index))
      this.selectTab(0)


    } catch (err) {
      console.log('Error:', err)
    }
  }

  async fetchAudioAndTextFiles() {
    try {
      const responseAudio = await fetch("/list-audio")
      const responseText = await fetch("/list-text")
      const audioFiles = await responseAudio.json()
      const textFiles = await responseText.json()
      console.log("Audio files:", audioFiles)
      console.log("Text files:", textFiles)
      return { audioFiles, textFiles }
    } catch (err) {
      console.error("Error fetching files:", err)
    }
  }

  async sendFolderSettings(audioFolder, textFolder) {
    try {
      const res = await fetch('/set-folders', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audioFolder, textFolder })
      })
      return res.text()
    } catch (err) {
      console.log('Error setting folders:', err)
    }
  }
}

// Event listener to set folders when button is clicked
document.getElementById('setFoldersButton').addEventListener('click', () => {
  const tabManager = new TabManager()
  tabManager.setFolders()
})

const tabManager = new TabManager()