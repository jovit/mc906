import React, {Component} from 'react';
import logo from './logo.svg';
import './App.css';
import { withStyles } from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import Slider from '@material-ui/lab/Slider'
import * as tf from '@tensorflow/tfjs';
import m from './model/model.json'

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
      values:[]
    }

    this.loadModel()
  }

  async loadModel() {
    const model = await tf.models.modelFromJSON(m);
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

    </div>
  );

}}

export default withStyles(styles)(App);;
