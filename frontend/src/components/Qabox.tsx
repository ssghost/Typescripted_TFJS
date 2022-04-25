import React, { useEffect, useState } from "react";
import { useRef } from "react";
import { AiFillEdit, AiFillDelete } from "react-icons/ai";
import { QA } from "../models/model";
import { Draggable } from "react-beautiful-dnd";


const Qabox: React.FC<{
  index: number;
  qa: QA;
  qas: Array<QA>;
  setQas: React.Dispatch<React.SetStateAction<Array<QA>>>;
}> = ({ index, qa, qas, setQas }) => {
  const [edit, setEdit] = useState<boolean>(false);
  const [editQa, setEditQa] = useState<string>(qa.que+"/n"+qa.ans);
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    inputRef.current?.focus();
  });

  const handleEdit = (e: React.FormEvent, id: number) => {
    e.preventDefault();
    setQas(
      qas.map((qa) => (qa.id === id ? { ...qa, todo: editQa } : qa))
    );
    setEdit(false);
  };

  const handleDelete = (id: number) => {
    setQas(qas.filter((qa:any) => qa.id !== id));
  };

  return (
    <Draggable draggableId={qa.id.toString()} index={index}>
      {(provided:any, snapshot:any) => (
        <form
          onSubmit={(e) => handleEdit(e, qa.id)}
          {...provided.draggableProps}
          {...provided.dragHandleProps}
          ref={provided.innerRef}
          className={`qas__single ${snapshot.isDragging ? "drag" : ""}`}
        >
            <input
              value={editQa}
              onChange={(e) => setEditQa(e.target.value)}
              className="qas__single--text"
              ref={inputRef}
            />
          <div>
            <span
              className="icon"
              onClick={() => {
                if (!edit) {
                  setEdit(!edit);
                }
              }}
            >
              <AiFillEdit />
            </span>
            <span className="icon" onClick={() => handleDelete(qa.id)}>
              <AiFillDelete />
            </span>
          </div>
        </form>
      )}
    </Draggable>
  );
};

export default Qabox;
