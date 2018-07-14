let x_value = [];
let y_value = [];
var polynomialType = "Linear";
var learningType = "SGD";
let m, b, a, c, d;
const learningRate = 0.09;
var optimizer = tf.train.sgd(learningRate);
let dragging = false;

function changeCanvas() {
  clear();
  polynomialType = document.getElementById("polynomialType").value;
  learningType = document.getElementById("learningType").value;
  if(learningType == "SGD") {
    optimizer = tf.train.sgd(learningRate);
  } else if(learningType == "Adam") {
    optimizer = tf.train.adam(learningRate);
  }
  x_value = [];
  y_value = [];
  console.log(polynomialType);
  console.log(learningType);
}

function setup() {
  // put setup code here
  var cnv = createCanvas(windowWidth - 80, windowHeight - 130);
  var x = (windowWidth - width) / 2;
  cnv.position(x);
  cnv.parent("p5Canvas");
  cnv.mousePressed(canvasMousePressed);
  cnv.mouseReleased(canvasMouseReleased);

  m = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  a = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));
}

function loss_func(pred, labels) {
  return pred.sub(labels).square().mean();
}

function predict(x_value) {
  const xs = tf.tensor1d(x_value);

  if(polynomialType == "Linear") {
    // y = mx + b
    return xs.mul(a).add(b);
  } else if (polynomialType == "Binomial") {
    // y = ax^2 + bx + c
    return xs.square().mul(a).add(b.mul(xs)).add(c);
  } else if (polynomialType == "Trinomial") {
    // y = ax^3 + bx^2 + cx + d
    return xs.pow(tf.scalar(3)).mul(a)
            .add(b.square().mul(xs))
            .add(c.mul(xs))
            .add(d);
  }
}

function windowResized() {
  resizeCanvas(windowWidth - 80, windowHeight - 80);
}

function canvasMousePressed() {
  dragging = true;
}
function canvasMouseReleased() {
  dragging = false;
}

function draw() {
  background(51);
  if(dragging) {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_value.push(x);
    y_value.push(y);
  } else {
    tf.tidy(() => {
      if (x_value.length > 0) {
        const ys = tf.tensor1d(y_value);
        optimizer.minimize(() => loss_func(predict(x_value), ys));
      }
    });
  }

  // for pointing pixels on the screen
  stroke(255);
  strokeWeight(13);
  for(let i = 0; i <= x_value.length; i++) {
    let px = map(x_value[i], -1, 1, 0, width);
    let py = map(y_value[i], -1, 1, height, 0);
    point(px, py);
  }

  if(polynomialType == "Linear") {
    let xs = [-1, 1];
    const ys = tf.tidy(() => predict(xs)); 

    let x1 = map(xs[0], -1, 1, 0, width);
    let x2 = map(xs[1], -1, 1, 0, width);
    let liney = ys.dataSync(); 
    ys.dispose();

    let y1 = map(liney[0], -1, 1, height, 0);
    let y2 = map(liney[1], -1, 1, height, 0);

    strokeWeight(2);
    line(x1, y1, x2, y2);
  } else if(polynomialType == "Binomial" || polynomialType == "Trinomial") {
    const curveXs = [];
    for(let i = -1; i <= 1.01; i += 0.02) {
      curveXs.push(i);
    }
    const ys = tf.tidy(() => predict(curveXs)); 
    let curveYs = ys.dataSync();
    ys.dispose();

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for (let i = 0; i < curveXs.length; i++) {
      let xs = map(curveXs[i], -1, 1, 0, width);
      let ys = map(curveYs[i], -1, 1, height, 0);
      vertex(xs, ys);
    }
    endShape();
  }
}