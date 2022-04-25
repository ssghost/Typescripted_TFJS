import React, { useState } from "react";
import "./App.css";
import InputField from "./components/InputField";
import BoxList from "./components/Boxlist";
import { DragDropContext, DropResult } from "react-beautiful-dnd";
import { QA } from "./models/model";
import format from "date-fns/format";
import * as tf from "@tensorflow/tfjs";


const App: React.FC = () => {
  const [que, setQue] = useState<string>("");
  const [qa, setQa] = useState<QA>({id:0, que:"", ans:""});
  const [qas, setQas] = useState<Array<QA>>([]);
  const now = Date.now()
  const tf_model_path = "../../model_"+format(now, 'YYYY-MM-DD HH:mm:ss')+"_converted"

  const handleAdd = (e: React.FormEvent) => {
    e.preventDefault();

    async function loadModel(){	
  	
      const tfmodel = await tf.loadLayersModel(tf_model_path);  

      return tfmodel
    }

    function predictModel(tfmodel:any, que:any){

      const ans:string = tfmodel.predict(que);

      return ans
    }

    if (que) {
      const tfmodel = loadModel();
      const ans = predictModel(tfmodel, que);
      setQa({id: Date.now(), que: que, ans: ans})
      setQas([...qas, qa]);
      setQue("");
    }
  };

  const onDragEnd = (result: DropResult) => {
    const { destination, source } = result;

    console.log(result);

    if (!destination) {
      return;
    }

    if (
      destination.droppableId === source.droppableId &&
      destination.index === source.index
    ) {
      return;
    }

    let add:QA;
    let active = qas;

    if (source.droppableId === "BoxList") {
      add = active[source.index];
      active.splice(source.index, 1, add);
    }     
  };

  return (
    <DragDropContext onDragEnd={onDragEnd}>
      <div className="App">
        <span className="heading">Question - Answer List</span>
        <InputField que={que} setQue={setQue} handleAdd={handleAdd} />
        <BoxList
          qas={qas}
          setQas={setQas}
        />
      </div>
    </DragDropContext>
  );
};

export default App;
