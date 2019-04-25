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
      values: Array.from({length: 50}, (x,i) => i).map(it => 0)
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

    console.log(values)

    console.log(this.model.predict([tf.tensor(values)]))
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

    </div>
  );

}}

export default withStyles(styles)(App);;
