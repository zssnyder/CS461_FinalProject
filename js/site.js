

var canvas = document.getElementById('myCanvas');
var canvasRect = canvas.getBoundingClientRect();
var ctx = canvas.getContext('2d');
var isDrawing;

canvas.onmousedown = function(e) {
  isDrawing = true;
  ctx.lineWidth = 10;
  ctx.lineJoin = ctx.lineCap = 'round';
  ctx.moveTo(e.pageX - canvasRect.left, e.pageY - canvasRect.y);
};

canvas.onmousemove = function(e) {
  if (isDrawing) {
    ctx.lineTo(e.pageX - canvasRect.left, e.pageY - canvasRect.y);
    ctx.stroke();
  }
};

canvas.onmouseup = function() {
  isDrawing = false;
};

function dlCanvas() {
    var dt = canvas.toDataURL('image/png');
    /* Change MIME type to trick the browser to downlaod the file instead of displaying it */
    dt = dt.replace(/^data:image\/[^;]*/, 'data:application/octet-stream');
  
    /* In addition to <a>'s "download" attribute, you can define HTTP-style headers */
    dt = dt.replace(/^data:application\/octet-stream/, 'data:application/octet-stream;headers=Content-Disposition%3A%20attachment%3B%20filename=Canvas.png');
  
    this.href = dt;
  };

  document.getElementById("dl").addEventListener('click', dlCanvas, false);
