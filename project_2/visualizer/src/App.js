import React, {Component} from 'react';
import logo from './logo.svg';
import './App.css';
import { withStyles } from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import Slider from '@material-ui/lab/Slider'
import * as tf from '@tensorflow/tfjs';
import m from './model/model.json'
import { Button } from '@material-ui/core';

const styles = {
  root: {
    width: 300,
  },
  slider: {
    padding: '22px 0px',
  },
};

class App extends Component {
  constructor(props) {
    super(props)

    this.state = {
      values: Array.from({length: 50}, (x,i) => 0)
    }

    this.loadModel = this.loadModel.bind(this);
    this.predict = this.predict.bind(this);

    this.loadModel()

  }

  async loadModel() {
    const model = await tf.loadLayersModel("https://raw.githubusercontent.com/jovit/mc906/master/project_2/visualizer/src/model/model.json");
    this.model = model;
  }

  predict() {
    const values = []
    for (const v of this.state.values) {
      values.push(v/100);
    }

    console.log(values)

    const pred = this.model.predict([tf.tensor(values, [1,50])])
    
    const readable_output = pred.dataSync();
    console.log(readable_output);


    var c = document.getElementById("myCanvas");
    var ctx = c.getContext("2d");

    var imgd = pred.dataSync().map(it => it * 255).map(it => Math.round(it));

    console.log(imgd)

    // first, create a new ImageData to contain our pixels
    var imgData = ctx.createImageData(28, 28); // width x height
    var data = imgData.data;

    // copy img byte-per-byte into our ImageData
    for (var i = 0; i < imgd.length; i++) {
        // red
        data[i*4] = imgd[i];
        // green
        data[(i*4)+1] = imgd[i];
        // blue
        data[(i*4)+2] = imgd[i];
        // alpha (always use max value)
        data[(i*4)+3] = 255;
    }

    // reset canvas scale
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    // use nearest neighboor for scaling (no blur)
    ctx.mozImageSmoothingEnabled = false;
    ctx.webkitImageSmoothingEnabled = false;
    // now we can draw our imagedata onto the canvas
    ctx.putImageData(imgData, 0, 0);
    // workaround to scale image data
    var imgPixels = new Image();
    // load canvas pixels
    imgPixels.src = c.toDataURL();
    imgPixels.onload = () => {
        // clear canvas
        ctx.clearRect(0, 0, c.width, c.height);
        // scale content
        ctx.scale(7.2, 7.2);
        // draw image
        ctx.drawImage(imgPixels, 0, 0);
    }
 }

  handleChange = (i) =>
    (event, value) => {
      console.log(i);
      console.log(value)
      const values = this.state.values
      values[i] = value
      this.setState({ values });
    };

 render(){ 
  const { classes } = this.props;

  return (
    <div className="App">

          {Array.from({length: 50}, (x,i) => i).map(i => <Slider
          key={i}
          classes={{ container: classes.slider }}
          value={this.state.values[i]}
          aria-labelledby="label"
          onChange={this.handleChange(i)}
          />)}

          <Button onClick={this.predict}>Predict</Button>

          <canvas width={200} height={200} id="myCanvas"  style={{border:"1px solid #000000"}}></canvas>

    </div>
  );

}}

export default withStyles(styles)(App);;
