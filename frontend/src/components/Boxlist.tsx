import React  from "react";
import {PropsWithChildren} from "react"
import { QA } from "../models/model";
import Qabox from "./Qabox";
import { Droppable } from "react-beautiful-dnd";

interface props {
  qas: Array<QA>;
  setQas: React.Dispatch<React.SetStateAction<Array<QA>>>;
}

const BoxList: React.FC<props> = ({
  qas,
  setQas
}:PropsWithChildren<any>) => {
  return (
    <div className="container">
      <Droppable droppableId="BoxList">
        {(provided:any, snapshot:any) => (
          <div
            className={`qas ${snapshot.isDraggingOver ? "dragactive" : ""}`}
            ref={provided.innerRef}
            {...provided.droppableProps}
          >
            <span className="qas__heading">Aswered Questions</span>
            {qas?.map((qa:any, index:any) => (
              <Qabox
                index={index}
                qas={qas}
                qa={qa}
                key={qa.id}
                setQas={setQas}
              />
            ))}
            {provided.placeholder}
          </div>
        )}
      </Droppable>
    </div>
  );
};

export default BoxList;
