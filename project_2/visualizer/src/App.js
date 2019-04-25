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
    for (const v in this.state.values) {
      values.push(v/100);
    }

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
        data[i] = imgd[i];
    }

    // now we can draw our imagedata onto the canvas
    ctx.putImageData(imgData, 0, 0);

    
  }

  handleChange = (i) =>
    (event, value) => {
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
