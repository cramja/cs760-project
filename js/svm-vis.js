// Helper functions
//

function shadeColor2(color, percent) {
    // see http://stackoverflow.com/questions/5560248/programmatically-lighten-or-darken-a-hex-color-or-rgb-and-blend-colors
    //
    var f=parseInt(color.slice(1),16),t=percent<0?0:255,p=percent<0?percent*-1:percent,R=f>>16,G=f>>8&0x00FF,B=f&0x0000FF;
    return "#"+(0x1000000+(Math.round((t-R)*p)+R)*0x10000+(Math.round((t-G)*p)+G)*0x100+(Math.round((t-B)*p)+B)).toString(16).slice(1);
}
function sign(i) {
    return i >= 0 ? 1 : -1;
}

var kSeed = 1;
function random() {
    // We implement our own random so that we can define the seed
    //
    var x = Math.sin(kSeed++) * 10000;
    return x - Math.floor(x);
}

function random_gaussian() {
    // returns 2 gaussian randomly distributed numbers with mean 1. Uses the Box-Muller algorithm
    // Also respects the global seed.
    //
    var x1, x2, w, y1, y2;
    do {
        x1 = 2.0 * random() - 1.0;
        x2 = 2.0 * random() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );

    w = Math.sqrt( (-2.0 * Math.log( w ) ) / w );
    y1 = x1 * w;
    y2 = x2 * w;
    return [y1, y2];
}

// Helper Classes
//

// Vector helper functions
//
function _scalarApply(vec, sc, fn) {
  var v = [];
  for (var i = 0; i < vec.length(); i++) {
    v.push(fn(vec.get(i),sc));
  }
  var rvec = new Vector(0);
  rvec.set(v);
  return rvec;
}

function _vectorApply(vec0, vec1, fn) {
  console.assert(vec0.length() == vec1.length(), 
      "vectors were unequal length in vector-wise apply")
  var v = [];
  for (var i = 0; i < vec0.length(); i++) {
    v.push(fn(vec0.get(i),vec1.get(i)));
  }
  var rvec = new Vector(0);
  rvec.set(v);
  return rvec;
}

function _apply(x,y,fn) {
  // x - assumed to be a vector
  // y - either a number or a vector
  // fn - the function to apply to x using data y
  if (typeof(y) == 'number') {
      return _scalarApply(x, y, fn);
  } else {
      return _vectorApply(x, y, fn);
  }
}

function Vector(length) {
    // A class for holding example data and weight vectors
    //

    this.data = [];
    for (var i = 0; i < length; i++) {
        this.data[i] = 0;
    }

    this.set = function(v) {
        this.data = v;
    }

    this.toList = function() {
        var l =[];
        for (var i = 0; i < this.length(); i++) {
            l.push(this.data[i]);
        }
        return l;
    };

    this.length = function() {
        return this.data.length;
    };

    this.toString = function() {
        var s = "[";
        for (var i in this.data) {
            s += this.data[i].toFixed(4) + ", ";
        }
        return s.substr(0,s.length-2) + "]";
    };

    this.norm = function() {
        // returns the l2 norm of the vector.
        //
        var sum = 0;
        for (var i = 0; i < this.length(); i++) {
            sum += Math.pow(this.data[i],2);
        }
        return Math.sqrt(sum);
    };

    this.dot = function(other) {
        // returns the dot product of the vectors
        //
        console.assert(other.length() == this.length(), "vectors were unequal length");
        var prod = 0;
        for (var i = 0; i < this.length(); i++) {
            prod += this.data[i] * other.data[i];
        }
        return prod;
    };

    this.multiply = function(other) {
        // returns the element wise multiplicative product
        //
        var fn = function(x,y){return x*y;}
        return _apply(this, other, fn);
    };

    this.divide = function(other) {
        // returns the element wise multiplicative product
        //
        var fn = function(x,y){return x/y;}
        return _apply(this, other, fn);
    };

    this.add = function (other) {
        // add
        //
        var fn = function(x,y){return x+y;}
        return _apply(this, other, fn);
    };

    this.sub = function (other) {
        // Vectorwise subtract
        //
        var fn = function(x,y){return x-y;}
        return _apply(this, other, fn);
    };

    this.randomize = function () {
      // Set every value in the vector to a random value.
      //
      for (var i = 0; i < this.length(); i++) {
          this.data[i] = Math.random();
      }
    }

    this.get = function(i) {
      return this.data[i];
    }
};

function DataSet(data, classification) {
    // Contains a list of vectors, as well as a Vector of the associated classifications.
    //

    this.data = data;
    this.classification = classification;
    this.idx = [];
    for (var i = 0; i < data.length; i++){
        this.idx[i] = i;
    }

    this.length = function() {
        return this.data.length;
    };

    this.get = function(i) {
        // returns a list of [coords, class] of the corresponding index of data.
        //
        var sidx = this.idx[i];
        console.assert(sidx < this.data.length, "error in reindexing");
        return [this.data[sidx], this.classification[sidx]];
    };

    this.shuffle = function() {
        // shuffles the data members
        //
        var i, j;
        for (i = this.data.length - 1; i > 0; i--) {
            j = Math.floor(random() * i);
            var t = this.idx[i];
            this.idx[i] = this.idx[j];
            this.idx[j] = t;
        }
    };

    this.padOnes = function() {
        // appends a bias term to each vector
        //
        for (var i = 0; i < this.length(); i++) {
            this.data[i].data.push(1);
        }
    };

    this.toString = function() {
        var s = "";
        for (var i = 0; i < this.data.length; i++) {
            s += (this.classification[i] == 1 ? " " : "") + this.classification[i] + " : "
                + this.data[i].toString() + "\n";
        }
        return s;
    };
}

function SVM(dataset, lambda, learning_rate) {
    this.data_set = dataset;
    this.learning_rate = learning_rate;
    this.lambda = lambda;

    this.kLearningRateDecay = 0.97;
    this.kColor = "#1b5886"

    this.next = function() {
        this.iter += 1;
        if (this.iter % this.data_set.length() == 0) {
            this.epoch += 1;
            this.data_set.shuffle();
            this.learning_rate_decayed = this.learning_rate_decayed * this.kLearningRateDecay;
        }
        var example = this.data_set.get(this.iter % this.data_set.length());
        var j = example[0].dot(this.w) * example[1];
        var dw = this.w.multiply(this.lambda * -1.0 /this.data_set.length());
        if (j < 1) {
            var grad_w = example[0].multiply(example[1]);
            dw = dw.add(grad_w);
        }
        dw = dw.multiply(this.learning_rate_decayed);
        this.w = this.w.add(dw);
    };

    this.setLearningRate = function(rate) {
        this.learning_rate = rate;
        this.learning_rate_decayed = rate;
    };

    this.accuracy = function() {
      var correct = 0.0;
      for (var i = 0; i < this.data_set.length(); i++) {
          var cls = this.data_set.classification[i];
          var dat = this.data_set.data[i];
          var pred = sign(this.w.dot(dat));
          if (pred == cls) {
              correct += 1;
          }
      }
      return correct / this.data_set.length();
    };

    this.reset = function() {
        this.w = new Vector(3);
        for (var i = 0; i < 3; i++) {
            this.w.data[i] = (random() - 0.5) * 10;
        }
        this.epoch = 0;
        this.iter = 0;
        this.learning_rate_decayed = this.learning_rate;
    };

    this.reset();
}

function SVM_asgd(dataset, lambda, learning_rate) {
    this.data_set = dataset;
    this.learning_rate = learning_rate;
    this.lambda = lambda;

    this.kLearningRateDecay = 0.97;
    this.kColor = "#138644"

    this.next = function() {
        this.iter += 1;
        if (this.iter % this.data_set.length() == 0) {
            this.epoch += 1;
            this.data_set.shuffle();
            this.learning_rate_decayed = this.learning_rate_decayed * this.kLearningRateDecay;
        }
        var example = this.data_set.get(this.iter % this.data_set.length());
        var j = example[0].dot(this.w) * example[1];
        var dw = this.w.multiply(this.lambda * -1.0 /this.data_set.length());
        if (j < 1) {
            var grad_w = example[0].multiply(example[1]);
            dw = dw.add(grad_w);
        }
        var mu = 1.0 / this.iter;
        this.w_avg = this.w_avg.multiply(1.0-mu).add(dw.multiply(mu));
        dw = dw.multiply(0.2).add(this.w_avg.multiply(0.8));
        dw = dw.multiply(this.learning_rate_decayed);
        this.w = this.w.add(dw);
    };

    this.setLearningRate = function(rate) {
        this.learning_rate = rate;
        this.learning_rate_decayed = rate;
    };

    this.accuracy = function() {
        var correct = 0.0;
        for (var i = 0; i < this.data_set.length(); i++) {
            var cls = this.data_set.classification[i];
            var dat = this.data_set.data[i];
            var pred = sign(this.w.dot(dat));
            if (pred == cls) {
                correct += 1;
            }
        }
        return correct / this.data_set.length();
    };

    this.reset = function() {
        this.epoch = 0;
        this.iter = 0;
        this.w = new Vector(3);
        this.w_avg = new Vector(3);
        for (var i = 0; i < 3; i++) {
            this.w.data[i] = (random() - 0.5) * 10;
            this.w_avg.data[i] = this.w.data[i];
        }
        this.learning_rate_decayed = this.learning_rate;
    };

    this.reset();
};

function SVM_saga(dataset, lambda, learning_rate) {
    this.data_set = dataset;
    this.learning_rate = learning_rate;
    this.lambda = lambda;

    this.kLearningRateDecay = 0.97;
    this.kColor = "#a71d5d"

    this.next = function() {
    // Do a stocastic update
    //
        this.iter += 1;
        if (this.iter % this.data_set.length() == 0) {
            this.epoch += 1;
            this.learning_rate_decayed = this.learning_rate_decayed * this.kLearningRateDecay;
            this.shuffle_idx();
        }
        var curr_idx = this.indices[this.iter % this.indices.length];
        var d_wsaga_p = this.w_dsaga[curr_idx];

        var example = this.data_set.get(curr_idx);
        var j = example[0].dot(this.w) * example[1];

        // Start by calculating the regularization term:
        var dw = this.w.multiply(this.lambda * -1.0 / this.data_set.length());
        if (j < 1) {
            var grad_w = example[0].multiply(example[1]);
            dw = dw.add(grad_w);
        }
        // now apply the SAGA averaging
        var saga_avg = this.w_dsaga_sum.divide(Math.min(this.iter, this.data_set.length()));
        var d_wsaga = dw.sub(d_wsaga_p).add(saga_avg);
        d_wsaga = d_wsaga.multiply(this.learning_rate_decayed);
        this.w = this.w.add(d_wsaga);

        this.w_dsaga_sum = this.w_dsaga_sum.sub(d_wsaga_p).add(d_wsaga);
        this.w_dsaga[curr_idx] = d_wsaga;
    };

    this.setLearningRate = function(rate) {
        this.learning_rate = rate;
        this.learning_rate_decayed = rate;
    };

    this.accuracy = function() {
        var correct = 0.0;
        for (var i = 0; i < this.data_set.length(); i++) {
            var cls = this.data_set.classification[i];
            var dat = this.data_set.data[i];
            var pred = sign(this.w.dot(dat));
            if (pred == cls) {
                correct += 1;
            }
        }
        return correct / this.data_set.length();
    };

    this.shuffle_idx = function() {
        var i, j;
        for (i = this.indices.length - 1; i > 0; i--) {
            j = Math.floor(random() * i);
            var t = this.indices[i];
            this.indices[i] = this.indices[j];
            this.indices[j] = t;
        }
    }

    this.reset = function() {
        this.epoch = 0;
        this.iter = 0;
        this.w = new Vector(3);
        this.w.randomize();
        this.w_dsaga = []; // Vector of past derivatives.
        this.indices = [];
        this.w_dsaga_sum = new Vector(3);
        for (var i = 0; i < this.data_set.length(); i++) {
            this.w_dsaga.push(new Vector(3));
            this.indices.push(i);
        }
        this.learning_rate_decayed = this.learning_rate;
    };

    this.reset();
};

function Plot(scale, data_set) {
    // Helper class to manage plotting functions. Does drawing on the canvas
    // center - center point of the Plot, usually 0,0
    // scale  - how many pixels there are per 1 unit square. A larger scale fits fewer points.
    // DataSet - the data being represented.
    //

    this.data_set = data_set;
    this.scale = scale;
    this.canvas = document.getElementById("svm_canvas");
    this.ctx = this.canvas.getContext("2d");
    this.height = this.canvas.height;
    this.width = this.canvas.width;

    this.kColorCenterLines = "#dedede";
    this.kColorNegCoords = "#FF3300";
    this.kColorPosCoords = "#0033FF";
    this.kBlack = "#000000";
    this.kFontColor = "#404040";
    this.kLineHeight = 30;
    this.kFontSize = 20;

    this.currentLine = 0;

    this.redraw = function() {
        this.ctx.clearRect(0,0,this.width,this.height);
        this.drawCenterLines();
        this.drawDataSet();
        this.currentLine = 0;
    };

    this.drawDataSet = function() {
        // Draws the points associated with the sample data.
        //
        for(var i = 0; i < this.data_set.data.length; i++) {
            var coords = this.data_set.data[i].data;
            var cls = this.data_set.classification[i];
            this.drawPointFromCoords(coords, cls == -1 ? this.kColorNegCoords : this.kColorPosCoords);
        }
    };

    this.drawCenterLines = function() {
        // Draws the central axis on the plot. We assume the plot is centered at 0,0
        //
        var midX = this.width/2.0;
        var midY = this.height/2.0;
        var line_color = this.kColorCenterLines;
        this.drawLine([0,midY],[this.width, midY],line_color);
        this.drawLine([midX,0],[midX, this.height],line_color);
    };

    this.drawPointFromCoords = function(pt, color) {
        // Draws a point on the plot given a point which is in relative coordinates.
        //
        var radius = 4.0;
        var pt_ = this.coordinateToPoint(pt);
        pt_[0] -= radius/2.0;
        pt_[1] -= radius/2.0;
        this.drawPoint(pt_, radius, color);
    };

    this.drawLineFromCoords = function(coord0, coord1, color) {
        // Draws a line to and from the given relative coordinates.
        //
        var pt0 = this.coordinateToPoint(coord0);
        var pt1 = this.coordinateToPoint(coord1);
        this.drawLine(pt0,pt1,color);
    };

    this.drawPoint = function(pt, radius, color) {
        // Draws a circle on the canvas. Pt is given in absolute coordinates, ie does not do
        // translation
        //
        this.ctx.beginPath();
        this.ctx.arc(pt[0], pt[1], radius, 0, 2 * Math.PI, false);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = this.kBlack;
        this.ctx.stroke();
    };

    this.drawLine = function(pt0, pt1, color) {
        // Draws a line on the canvas. Does not do translation
        //
        this.ctx.beginPath();
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = color;
        this.ctx.moveTo(pt0[0],pt0[1]);
        this.ctx.lineTo(pt1[0],pt1[1]);
        this.ctx.stroke();
    };

    this.writeText = function(text) {
        // Draws text on the canvas. Draws it on the next logical line.
        //
        this.ctx.fillStyle = this.kFontColor;
        this.drawText([10, this.kLineHeight * (this.currentLine + 1)],text);
        this.currentLine += 1;
    };

    this.drawText = function(pt, text) {
        // Draws text on the canvas. Does not do translation
        //
        this.ctx.fillStyle = this.kFontColor;
        this.ctx.font = this.kFontSize + "px Monospace";
        this.ctx.fillText(text,pt[0],pt[1]);
    };

    this.coordinateToPoint = function(pt) {
        // translates a point to a coordinate pair that can be drawn on the graph
        // returns a coordinate [x,y]
        //
        var cX = this.width/2.0;
        var cY = this.height/2.0;
        return [cX + pt[0] * this.scale, cY - pt[1] * this.scale];
    };
}

function createDataSet(num_examples, center0, center1, mean) {
    // Creates a set of x,y points. Points are centered at 2,2 and -2,-2 with classes of 1 and -1.
    //
    var data_instances = [];
    var data_classes = [];
    for (var i = 0; i < num_examples; i++) {
        var rn = random_gaussian();
        var v = new Vector(2);
        if (i % 2) {
            v.data[0] = center0[0] + rn[0] * mean;
            v.data[1] = center0[1] + rn[1] * mean;
        } else {
            v.data[0] = center1[0] + rn[0] * mean;
            v.data[1] = center1[1] + rn[1] * mean;
        }
        data_instances.push(v);
        data_classes.push(i % 2 ? -1 : 1);
    }
    var ds = new DataSet(data_instances, data_classes);
    ds.padOnes();
    return ds;
}

function Engine() {
    // Class which controls the simulation's state and handles interactivity
    //

    this.kNumExamples = 40;
    this.kDataMean = 1.2;
    this.kCenter0 = [-2,-2];
    this.kCenter1 = [2,2];
    this.kPaused = true;

    this.data_set = createDataSet(this.kNumExamples,this.kCenter0, this.kCenter1,this.kDataMean);
    this.plot = new Plot(30.0, this.data_set);
    this.updates = 0;

    this.solver_names = ["solver_sgd","solver_asgd","solver_saga"];
    this.solvers = {
        "solver_sgd" : new SVM(this.data_set, 1.0, 0.05),
        "solver_asgd" : new SVM_asgd(this.data_set, 1.0, 0.05),
        "solver_saga" : new SVM_saga(this.data_set, 1.0, 0.05)
    };
    for (var i in this.solver_names) {
        this.solvers[this.solver_names[i]].enabled = (i == 0);
    }

    this.reset = function() {
        this.kPaused = true;
        this.data_set = createDataSet(
            this.kNumExamples,
            this.kCenter0,
            this.kCenter1,
            this.kDataMean);
        this.plot.data_set = this.data_set;
        this.plot.redraw();
        this.updates = 0;

        for (var i in this.solver_names) {
            this.solvers[this.solver_names[i]].data_set = this.data_set;
            this.solvers[this.solver_names[i]].reset();
        }
    };

    this.wVecToCoords = function(w) {
        // Takes a vec of x,y,bias and transforms it into 2 points representing the vector.
        // Useful when drawing the margins. Returns a list of 2 points (which are also lists).
        //
        var kBound = 50;
        return [[-kBound, (-1.0 * w[2] + kBound * w[0]) / w[1]],
                [kBound, (-1.0 * w[2] - kBound * w[0]) / w[1]]];
    };

    this.drawDecisionBoundaryWithMargin = function(w, color0, color1) {
        // Draws a decision boundary using the weight vector provided
        //
        var bound = this.wVecToCoords(w.toList());
        this.plot.drawLineFromCoords(bound[0], bound[1], color0);
        var ang = Math.atan2(bound[0][1] - bound[1][1], bound[0][0] - bound[1][0]);
        var applyOffset = function(angle, magnitude, pt) {
            return [Math.cos(angle) * magnitude + pt[0],
                Math.sin(angle) * magnitude + pt[1]];
        };
        var margin_offset = 1/w.norm();
        var ang_minus = ang + (Math.PI/2.0);
        this.plot.drawLineFromCoords(
            applyOffset(ang_minus, margin_offset, bound[0]),
            applyOffset(ang_minus, margin_offset, bound[1]),
            color1);

        var ang_plus = ang - (Math.PI/2.0);
        this.plot.drawLineFromCoords(
            applyOffset(ang_plus, margin_offset, bound[0]),
            applyOffset(ang_plus, margin_offset, bound[1]),
            color1);
    };

    this.animate = function() {
        if (!this.kPaused) {
            // do redraw loop
            this.plot.redraw();
            for (var i in this.solver_names) {
                var solver = this.solvers[this.solver_names[i]];
                if (solver.enabled) {
                    solver.next();
                    var color_bound = solver.kColor;
                    this.drawDecisionBoundaryWithMargin(solver.w, color_bound, shadeColor2(color_bound, 0.8));
                }
            }
            this.updates += 1;
            this.plot.writeText("e: " + (Math.floor(this.updates/this.data_set.length())));
            this.plot.writeText("i: " + this.updates);
        }

    };

    // this initialization stuff feels like a hack:
    this.reset();
}

var engine = new Engine();
function animationLoop() {
    engine.animate();
    setTimeout(animationLoop, 10);
}

// set up all of the control elements
//

function resetEngineAndControls() {
    kSeed = 1;
    engine.reset();
    playpause_btn.innerHTML = "Play";
}

// Animation control
//
var playpause_btn = document.getElementById("playpause_btn");
playpause_btn.onclick = function(ev){
    engine.kPaused = !engine.kPaused;
    playpause_btn.innerHTML = engine.kPaused ? "Play" : "Pause";
};
var reset_btn = document.getElementById("reset_btn");
reset_btn.onclick = function(ev) {
    resetEngineAndControls();
};

// Data Control
//
var data_mean_slider = document.getElementById("data_mean_slider");
data_mean_slider.onchange = function(ev) {
    var v = this.value / 20;
    engine.kDataMean = v;
    document.getElementById("data_mean_value").innerHTML = "" + v;
    resetEngineAndControls();
};

// Solver Control
//

function toggle_solver(solver, enable) {
    // resets the engine and enables or disables a particular solver.
    //
    resetEngineAndControls();
    engine.solvers[solver].enabled = enable;
}

var learning_rates = [0.001, 0.01, 0.1, 0.5, 1, 2, 4];
var lambda_values = [0.001, 0.01, 0.1, 1, 2, 4, 8];
for (var solver_name_i in engine.solver_names) {
    var name = engine.solver_names[solver_name_i];
    var enabled_chk = document.getElementById(name + ".enabled-chk");
    enabled_chk.checked = engine.solvers[name].enabled;
    enabled_chk.onclick = (function(e) {toggle_solver(this, e.target.checked)}).bind(name);

    new Slider(
        name + ".learning_rate",
        learning_rates,
        (function (v){engine.solvers[this].setLearningRate(parseFloat(v)); }).bind(name));

    new Slider(
        name + ".lambda",
        lambda_values,
        (function (v){engine.solvers[this].lambda = parseFloat(v);}).bind(name));

    var c_div = document.getElementById(name);
    c_div.style.backgroundColor = shadeColor2(engine.solvers[name].kColor, 0.8);
}

animationLoop();
