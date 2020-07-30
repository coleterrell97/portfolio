var cells = document.querySelectorAll("td");
var reset = document.querySelector("#reset");


//-----------------------------------------------------------------------
//Users can change the number in a cell with a single click. Number wraps back to empty after "9"
function changeNumber(){
  if(this.textContent === " "){
    this.textContent = "1";
  }
  else if(this.textContent === "9"){
    this.textContent = " ";
  }
  else{
    this.textContent = parseInt(this.textContent) + 1
  }
}

//Adding event listeners to each cell of the game board
for(var i = 0; i < cells.length; i++){
  cells[i].addEventListener("click", changeNumber);
}

//-----------------------------------------------------------------------
//add event listener for reset button
reset.addEventListener("click", function(){
  for(var i = 0; i < cells.length; i ++){
    cells[i].textContent = " ";
  }
});
