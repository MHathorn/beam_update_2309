var http = new XMLHttpRequest();

let inputImages = [];

async function loadInput() {
  console.log("Entering the file upload section.");
  let entry = await Neutralino.os.showFolderDialog("Select image directory");
  console.log("You have selected:", entry);

  const url = `http://127.0.0.1:8005/uploadImages/?folder=${entry}`;
  
  
  
  
  http.open("GET", url, true);

  //Send the proper header information along with the request
  //http.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');

  http.onload = function () {
    if (http.status == 200) {
      document.getElementById("inputLabel").innerHTML =
        "Folder Successfully selected. Proceed to Model selection.";

      console.log(JSON.parse(http.response).selectedImages);
      inputImages = JSON.parse(http.response).selectedImages;
      document.getElementById('button1').classList.remove('rejected');
      document.getElementById('button1').classList.add('complete');
    } else
      {document.getElementById("inputLabel").innerHTML =
        "The selected folder does not contain any images. Select folder with tiff/tif images.";
      document.getElementById('button1').classList.add('rejected');}
  };
  http.send();
}

async function loadModel(init) {
  console.log("Entering the model upload section.");
  console.log(init)
  let entries = []
  let showText = ""
  if (init === false){
    entries = await Neutralino.os.showOpenDialog("Select Building Detection Model", {
      filters: [{ name: "Model Files", extensions: ["pkl"] }],
    });
    showText = "Model successfully selected. Proceed to input folder selection.";
  }
  else{
    entries = ['../models/HRNet.pkl']
    showText = "Default model successfully selected. Proceed to input folder selection.";
  }
  console.log("You have selected:", entries[0]);

  const url = `http://127.0.0.1:8005/loadModel/?model=${entries[0]}`;
  http.open("GET", url, true);

  //Send the proper header information along with the request
  //http.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');

  http.onload = function () {
    if (http.status == 200) {
      document.getElementById("inputLabel").innerHTML = showText;
      console.log(JSON.parse(http.response).selectedModel);
      document.getElementById('button2').classList.add('complete');
    } else
      {document.getElementById("inputLabel").innerHTML =
        "Invalid model file. Please select a model with .pkl extension.";
      document.getElementById('button2').classList.add('rejected');}
  };
  http.send();
}

async function loadOutput() {
  console.log("Entering the output folder section.");
  let entry = await Neutralino.os.showFolderDialog("Select output directory");
  console.log("You have selected:", entry);

  const url = `http://127.0.0.1:8005/loadOutputDir/?folder=${entry}`;
  http.open("GET", url, true);

  http.onload = function () {
    if (http.status == 200) {
      document.getElementById("inputLabel").innerHTML =
        "Folder successfully selected. Proceed to building detection step.";
      console.log(JSON.parse(http.response).selectedImages);
      document.getElementById('button3').classList.add('complete');
    } else
      {document.getElementById("inputLabel").innerHTML =
        "Folder does not exist. Please select a valid output folder.";
      document.getElementById('button3').classList.add('rejected');}
  };
  http.send();
}



async function requestInferencing(inputImage, index, length) {
  return new Promise(function (resolve, reject) {
    const url = `http://127.0.0.1:8005/startInference/?file=${inputImage}`;
    http.open("GET", url, true);
    document.getElementById(
      "inputLabel"
    ).innerHTML = `Started building detection process. Detecting ${
      index + 1
    } of ${length} images`;
    http.onload = function () {
      if (http.status == 200) {
        document
          .getElementById("progress")
          .style.setProperty(
            "--percent",
            (((index + 1) / length) * 100).toString() + "%"
          );
        resolve(http.response);
        console.log(JSON.parse(http.response).inferencedImage);
      } else
        {document.getElementById("inputLabel").innerHTML =
          "Could not detect buildings. Please restart the process.";
        document.getElementById('button4').classList.remove('inprogress');
        document.getElementById('button4').classList.add('rejected');}
      reject({
        status: http.status,
      });
    };
    http.send();
  });
}

async function startInference() {
  console.log("Entering the tiling section.");
  document.getElementById("inputLabel").innerHTML =
    "Started building detection. Progress will be displayed here.";
  document
    .getElementById("progress")
    .style.setProperty("visibility", "visible");


  let elapsedTime = document.getElementById("elapsedTime")
  elapsedTime.style.setProperty("visibility", "visible");
  elapsedTime.innerHTML = 'Elapsed time: calculating...'


  let inferenceTime = document.getElementById("remainingTime")
  inferenceTime.innerHTML = 'Remaining time: calculating...'
  inferenceTime.style.setProperty("visibility", "visible");


  let ul = document.getElementById("imageList");
  ul.innerHTML = '';


  for (let index = 0; index < inputImages.length; index++) {
    let li = document.createElement("li");
    li.setAttribute("id", "item" + index.toString());
    li.appendChild(document.createTextNode(inputImages[index]));
    ul.appendChild(li);
  }

  function transformSeconds(totalSeconds){
    let hours = Math.floor(totalSeconds / 3600);
    totalSeconds %= 3600;
    let minutes = Math.floor(totalSeconds / 60);
    let seconds = Math.round(totalSeconds % 60);
    minutes = String(minutes).padStart(2, "0");
    hours = String(hours).padStart(2, "0");
    seconds = String(seconds).padStart(2, "0");
    return {hours: hours, minutes: minutes, seconds: seconds}

  }

  let totalStart = Date.now();

  let elapsedTimeIntervalRef = setInterval(() => {
    displayedElapsedTime = Date.now()
    let totalSeconds = (displayedElapsedTime - totalStart) / 1000;
    // console.log(new Date(totalSeconds * 1000).toISOString())
    time = transformSeconds(totalSeconds);
    // console.log(time)
    elapsedTime.innerHTML = `Elapsed time: ${time.hours}:${time.minutes}:${time.seconds}`
    }, 1000);

  document.getElementById('button4').classList.add('inprogress');
  for (let index = 0; index < inputImages.length; index++) {
    var li = document.getElementById("item" + index.toString());
    li.innerHTML = inputImages[index] + " : Building detection in progress.";
    let start = Date.now();
    await requestInferencing(inputImages[index], index, inputImages.length);
    let stop = Date.now()
    let totalSingleTime = Math.floor((stop-start) / 1000) // in seconds
    let remainingTime = transformSeconds(totalSingleTime * (inputImages.length - index))
    if (index!=0)
      inferenceTime.innerHTML = `Remaining time: ${remainingTime.hours}:${remainingTime.minutes}:${remainingTime.seconds}`
    li.innerHTML = inputImages[index] + " : Building detection finished.";
  }

  clearInterval(elapsedTimeIntervalRef)
  elapsedTime.style.setProperty("visibility", "visible");
  inferenceTime.innerHTML = `Remaining time: 00:00:00`



  document.getElementById("inputLabel").innerHTML =
    "Building detection completed. Shapefiles were stored in the output folder and can be validated now.";
  document.getElementById('button4').classList.remove('inprogress');
  document.getElementById('button4').classList.add('complete');
}

function openDocs() {
  console.log("Opening Documents");
  Neutralino.os.open("https://neutralino.js.org/docs");
}

function setTray() {
  if (NL_MODE != "window") {
    console.log("INFO: Tray menu is only available in the window mode.");
    return;
  }
  let tray = {
    icon: "/resources/icons/trayIcon.png",
    menuItems: [
      { id: "VERSION", text: "Get version" },
      { id: "SEP", text: "-" },
      { id: "QUIT", text: "Quit" },
    ],
  };
  Neutralino.os.setTray(tray);
}

function onTrayMenuItemClicked(event) {
  switch (event.detail.id) {
    case "VERSION":
      Neutralino.os.showMessageBox(
        "Version information",
        `Neutralinojs server: v${NL_VERSION} | Neutralinojs client: v${NL_CVERSION}`
      );
      break;
    case "QUIT":
      Neutralino.app.exit();
      break;
  }
}


function exitBackend() {
  const url = `http://127.0.0.1:8005/exit`;
  http.open("POST", url, true);
  http.send()
}


function onWindowClose() {
  exitBackend();
  Neutralino.app.exit();
}

function checkBackend() {
  const url = `http://localhost:8005/ping`;
  document.getElementById(
    "inputLabel"
  ).innerHTML = `Loading default model. Please wait.`;
  // repeat with the interval of 2 seconds
  let timerId = setInterval(() => {
    http.open("GET", url, true);
    http.onload = function () {
      if (http.status == 200) {
        document.getElementById(
          "inputLabel"
        ).innerHTML = `Default model loaded. Continue with the selection of the input folder directory.`;
        console.log(JSON.parse(http.response).started);
        clearInterval(timerId);
        loadModel(true);
        
      }
    };
    http.send();
  }, 1000);

  setTimeout(() => { clearInterval(timerId); }, 100000);
  
}





Neutralino.init();

Neutralino.events.on("ready", ()=>{console.log('Starting Processes'); checkBackend(); });


Neutralino.events.on("trayMenuItemClicked", onTrayMenuItemClicked);
Neutralino.events.on("windowClose", () =>{
  exitBackend();
  onWindowClose();
} );

if (NL_OS != "Darwin") {
  // TODO: Fix https://github.com/neutralinojs/neutralinojs/issues/615
  setTray();
}

showInfo();
