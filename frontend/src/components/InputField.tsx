import React, { PropsWithChildren, useRef } from "react";
import "./styles.css";

interface props {
  que: string;
  setQue: React.Dispatch<React.SetStateAction<string>>;
  handleAdd: (e: React.FormEvent) => void;
}

const InputField: React.FC<props> = ({ que, setQue, handleAdd }:PropsWithChildren<any>) => {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <form
      className="input"
      onSubmit={(e) => {
        handleAdd(e);
        inputRef.current?.blur();
      }}
    >
      <input
        type="text"
        placeholder="Enter a Question."
        value={que}
        ref={inputRef}
        onChange={(e) => setQue(e.target.value)}
        className="input_box"
      />
      <button type="submit" className="input_submit">
        ANSWER
      </button>
    </form>
  );
};

export default InputField;
