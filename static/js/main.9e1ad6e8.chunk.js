(window.webpackJsonp=window.webpackJsonp||[]).push([[0],{247:function(e,t,n){e.exports=n(427)},252:function(e,t,n){},254:function(e,t,n){e.exports=n.p+"static/media/logo.5d5d9eef.svg"},255:function(e,t,n){},259:function(e,t){},260:function(e,t){},262:function(e,t){},295:function(e,t){},296:function(e,t){},342:function(e,t){},343:function(e,t){},344:function(e,t){},345:function(e){e.exports={}},427:function(e,t,n){"use strict";n.r(t);var a=n(4),o=n.n(a),r=n(30),c=n.n(r),i=(n(252),n(75)),l=n.n(i),s=n(136),u=n(137),d=n(138),h=n(143),m=n(139),f=n(40),v=n(144),p=(n(254),n(255),n(16)),g=n(140),b=n.n(g),w=n(76),y=(n(345),n(142)),j=function(e){function t(e){var n;return Object(u.a)(this,t),(n=Object(h.a)(this,Object(m.a)(t).call(this,e))).handleChange=function(e){return function(t,a){console.log(e),console.log(a);var o=n.state.values;o[e]=a,n.setState({values:o})}},n.state={values:Array.from({length:100},function(e,t){return 0})},n.loadModel=n.loadModel.bind(Object(f.a)(n)),n.predict=n.predict.bind(Object(f.a)(n)),n.loadModel(),n}return Object(v.a)(t,e),Object(d.a)(t,[{key:"loadModel",value:function(){var e=Object(s.a)(l.a.mark(function e(){var t;return l.a.wrap(function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,w.a("https://raw.githubusercontent.com/jovit/mc906/master/project_2/visualizer/src/model/model.json");case 2:t=e.sent,this.model=t;case 4:case"end":return e.stop()}},e,this)}));return function(){return e.apply(this,arguments)}}()},{key:"predict",value:function(){var e=[],t=!0,n=!1,a=void 0;try{for(var o,r=this.state.values[Symbol.iterator]();!(t=(o=r.next()).done);t=!0){var c=o.value;e.push(c/100)}}catch(p){n=!0,a=p}finally{try{t||null==r.return||r.return()}finally{if(n)throw a}}console.log(e);var i=this.model.predict([w.b(e,[1,100])]),l=i.dataSync();console.log(l);var s=document.getElementById("myCanvas"),u=s.getContext("2d"),d=i.dataSync().map(function(e){return 255*e}).map(function(e){return Math.round(e)});console.log(d);for(var h=u.createImageData(50,50),m=h.data,f=0;f<d.length;f++)m[4*f]=d[f],m[4*f+1]=d[f],m[4*f+2]=d[f],m[4*f+3]=255;u.setTransform(1,0,0,1,0,0),u.mozImageSmoothingEnabled=!1,u.webkitImageSmoothingEnabled=!1,u.putImageData(h,0,0);var v=new Image;v.src=s.toDataURL(),v.onload=function(){u.clearRect(0,0,s.width,s.height),u.scale(5,5),u.drawImage(v,0,0)}}},{key:"render",value:function(){var e=this,t=this.props.classes;return o.a.createElement("div",{className:"App"},Array.from({length:100},function(e,t){return t}).map(function(n){return o.a.createElement(b.a,{key:n,classes:{container:t.slider},value:e.state.values[n],"aria-labelledby":"label",onChange:e.handleChange(n)})}),o.a.createElement(y.a,{onClick:this.predict},"Predict"),o.a.createElement("canvas",{width:200,height:200,id:"myCanvas",style:{border:"1px solid #000000"}}))}}]),t}(a.Component),k=Object(p.withStyles)({root:{width:300},slider:{padding:"22px 0px"}})(j);Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));c.a.render(o.a.createElement(k,null),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then(function(e){e.unregister()})}},[[247,1,2]]]);
//# sourceMappingURL=main.9e1ad6e8.chunk.js.map